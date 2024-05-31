use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex, OnceLock},
    thread,
};

use alloy_rpc_types::{Block, Header};
use revm::primitives::{Account, AccountInfo, BlockEnv, ResultAndState, SpecId, TxEnv};

use crate::{
    mv_memory::MvMemory,
    primitives::{get_block_env, get_block_spec, get_tx_envs},
    scheduler::Scheduler,
    vm::{ExecutionError, Vm, VmExecutionResult},
    ExecutionTask, MemoryLocation, MemoryValue, Storage, Task, TxVersion, ValidationTask,
};

/// Errors when executing a block with PEVM.
#[derive(Debug)]
pub enum PevmError {
    /// Cannot derive the chain spec from the block header.
    UnknownBlockSpec,
    /// Block header lacks information for execution.
    MissingHeaderData,
    /// Transactions lack information for execution.
    MissingTransactionData,
    /// EVM execution error.
    ExecutionError(ExecutionError),
    /// Impractical errors that should be unreachable.
    /// The library has bugs if this is yielded.
    UnreachableError,
}

/// Execution result of PEVM.
pub type PevmResult = Result<Vec<ResultAndState>, PevmError>;

/// Parallel Executor
#[derive(Debug, Default)]
pub struct Pevm {
    mv_memory: Arc<MvMemory>,
    scheduler: Scheduler,
    execution_results: Vec<Mutex<Option<ResultAndState>>>,
    execution_error: OnceLock<ExecutionError>,
}

impl Pevm {
    /// Execute an Alloy block, which is becoming the "standard" format in Rust.
    /// TODO: Better error handling.
    pub fn execute<S: Storage + Send + Sync>(
        &mut self,
        storage: S,
        block: Block,
        parent_header: Option<Header>,
        concurrency_level: NonZeroUsize,
    ) -> PevmResult {
        let Some(spec_id) = get_block_spec(&block.header) else {
            return Err(PevmError::UnknownBlockSpec);
        };
        let Some(block_env) = get_block_env(&block.header, parent_header.as_ref()) else {
            return Err(PevmError::MissingHeaderData);
        };
        let Some(tx_envs) = get_tx_envs(&block.transactions) else {
            return Err(PevmError::MissingTransactionData);
        };
        self.execute_revm(storage, spec_id, block_env, tx_envs, concurrency_level)
    }

    /// Execute an REVM block.
    // TODO: Better error handling.
    pub fn execute_revm<S: Storage + Send + Sync>(
        &mut self,
        storage: S,
        spec_id: SpecId,
        block_env: BlockEnv,
        txs: Vec<TxEnv>,
        concurrency_level: NonZeroUsize,
    ) -> PevmResult {
        if txs.is_empty() {
            return PevmResult::Ok(Vec::new());
        }

        // Beneficiary setup for post-processing
        let beneficiary_address = block_env.coinbase;
        let mut beneficiary_account_info = match storage.basic(beneficiary_address) {
            Ok(Some(account)) => account.into(),
            _ => AccountInfo::default(),
        };

        // Preparing the main components
        let block_size = txs.len();
        Arc::get_mut(&mut self.mv_memory)
            .unwrap() // TODO: Better error handling
            .prepare(block_size, MemoryLocation::Basic(beneficiary_address));

        // Edge case that is pretty common
        // TODO: Shortcut even before preparing the main parallel components.
        // It would require structuring a cleaner trait interface for any Storage
        // to act as the standalone DB for sequential execution.
        if block_size == 1 {
            let vm = Vm::new(spec_id, block_env, txs, storage, self.mv_memory.clone());
            return match vm.execute(0) {
                VmExecutionResult::ExecutionError(err) => Err(PevmError::ExecutionError(err)),
                VmExecutionResult::Ok {
                    mut result_and_state,
                    write_set,
                    ..
                } => {
                    for (location, value) in write_set {
                        if location == MemoryLocation::Basic(beneficiary_address) {
                            result_and_state.state.insert(
                                beneficiary_address,
                                post_process_beneficiary(&mut beneficiary_account_info, value),
                            );
                            break;
                        }
                    }
                    Ok(vec![result_and_state])
                }
                _ => Err(PevmError::UnreachableError),
            };
        }

        let max_concurrency_level = self.scheduler.prepare(&beneficiary_address, &txs);
        // TODO: Bring this to self? For now, it would require some type level friction, and
        // we cannot reuse much allocated VM memory anyway.
        let vm = Vm::new(spec_id, block_env, txs, storage, self.mv_memory.clone());
        // TODO: while loop is cleaner but slower? Or does the compiler optimize that well?
        for _ in self.execution_results.len()..block_size {
            self.execution_results.push(Mutex::default());
        }
        // End of preparing the main components

        // TODO: Better thread handling. Rayon thread pools are somehow significantly slower!
        thread::scope(|scope| {
            for _ in 0..concurrency_level.min(max_concurrency_level).into() {
                scope.spawn(|| {
                    let mut task = None; // TODO: Get the first task right away?
                    let mut consecutive_empty_tasks: u8 = 0;
                    while !self.scheduler.done() {
                        // TODO: Have different functions or an enum for the caller to choose
                        // the handling behaviour when a transaction's EVM execution fails.
                        // Parallel block builders would like to exclude such transaction,
                        // verifiers may want to exit early to save CPU cycles, while testers
                        // may want to collect all execution results. We are exiting early as
                        // the default behaviour for now.
                        if self.execution_error.get().is_some() {
                            break;
                        }

                        // Find and perform the next execution or validation task.
                        //
                        // After an incarnation executes it needs to pass validation. The
                        // validation re-reads the read-set and compares the observed versions.
                        // A successful validation implies that the applied writes are still
                        // up-to-date. A failed validation means the incarnation must be
                        // aborted and the transaction is re-executed in a next one.
                        //
                        // A successful validation does not guarantee that an incarnation can be
                        // committed. Since an abortion and re-execution of an earlier transaction
                        // in the block might invalidate the incarnation read set and necessitate
                        // re-execution. Thus, when a transaction aborts, all higher transactions
                        // are scheduled for re-validation. The same incarnation may be validated
                        // multiple times, by different threads, and potentially in parallel, but
                        // BlockSTM ensures that only the first abort per version succeeds.
                        //
                        // Since transactions must be committed in order, BlockSTM prioritizes
                        // tasks associated with lower-indexed transactions.
                        task = match task {
                            Some(Task::Execution(tx_version)) => {
                                self.try_execute(&vm, tx_version).map(Task::Validation)
                            }
                            Some(Task::Validation(tx_version)) => {
                                self.try_validate(&tx_version).map(Task::Execution)
                            }
                            None => self.scheduler.next_task(),
                        };

                        if task.is_none() {
                            consecutive_empty_tasks += 1;
                        } else {
                            consecutive_empty_tasks = 0;
                        }
                        // Many consecutive empty tasks usually mean the number of remaining tasks
                        // is smaller than the number of threads, or they are highly sequential
                        // anyway. This early exit helps remove thread overheads and join faster.
                        if consecutive_empty_tasks == 3 {
                            break;
                        }
                    }
                });
            }
        });

        if let Some(err) = self.execution_error.take() {
            return Err(PevmError::ExecutionError(err));
        }

        // We lazily evaluate the final beneficiary account's balance at the end of each transaction
        // to avoid "implicit" dependency among consecutive transactions that read & write there.
        // TODO: Refactor, improve speed & error handling.
        let beneficiary_values = self.mv_memory.consume_beneficiary();
        Ok(self
            .execution_results
            .iter()
            .zip(beneficiary_values)
            .map(|(mutex, value)| {
                let mut result_and_state = mutex.lock().unwrap().take().unwrap();
                result_and_state.state.insert(
                    beneficiary_address,
                    post_process_beneficiary(&mut beneficiary_account_info, value),
                );
                result_and_state
            })
            .collect())
    }

    // Execute the next incarnation of a transaction.
    // If an ESTIMATE is read, abort and add dependency for re-execution.
    // Otherwise:
    // - If there is a write to a memory location to which the
    //   previous finished incarnation has not written, create validation
    //   tasks for all higher transactions.
    // - Otherwise, return a validation task for the transaction.
    fn try_execute<S: Storage>(&self, vm: &Vm<S>, tx_version: TxVersion) -> Option<ValidationTask> {
        match vm.execute(tx_version.tx_idx) {
            VmExecutionResult::ReadError { blocking_tx_idx } => {
                if !self
                    .scheduler
                    .add_dependency(tx_version.tx_idx, blocking_tx_idx)
                {
                    // Retry the execution immediately if the blocking transaction was
                    // re-executed by the time we can add it as a dependency.
                    return self.try_execute(vm, tx_version);
                }
                None
            }
            VmExecutionResult::ExecutionError(err) => {
                // TODO: Better error handling
                self.execution_error.set(err).unwrap();
                None
            }
            VmExecutionResult::Ok {
                result_and_state,
                read_set,
                write_set,
            } => {
                *index_mutex!(self.execution_results, tx_version.tx_idx) = Some(result_and_state);
                let wrote_new_location = self.mv_memory.record(&tx_version, read_set, write_set);
                self.scheduler
                    .finish_execution(tx_version, wrote_new_location)
            }
        }
    }

    // Validate the last incarnation of the transaction.
    // If validation fails:
    // - Mark every memory value written by the incarnation as ESTIMATE.
    // - Create validation tasks for all higher transactions that have
    //   not been executed.
    // - Return a re-execution task for this transaction with an incremented
    //   incarnation.
    fn try_validate(&self, tx_version: &TxVersion) -> Option<ExecutionTask> {
        let read_set_valid = self.mv_memory.validate_read_set(tx_version.tx_idx);
        let aborted = !read_set_valid && self.scheduler.try_validation_abort(tx_version);
        if aborted {
            self.mv_memory
                .convert_writes_to_estimates(tx_version.tx_idx);
        }
        self.scheduler.finish_validation(tx_version, aborted)
    }
}

// Fully evaluate a beneficiary account at the end of block execution,
// including lazy updating atomic balances.
// TODO: Cleaner interface, better code location & error handling
fn post_process_beneficiary(
    beneficiary_account_info: &mut AccountInfo,
    value: MemoryValue,
) -> Account {
    match value {
        MemoryValue::Basic(info) => {
            *beneficiary_account_info = *info;
        }
        MemoryValue::LazyBeneficiaryBalance(addition) => {
            beneficiary_account_info.balance += addition;
        }
        _ => unreachable!(),
    }
    // TODO: This potentially wipes beneficiary account's storage.
    // Does that happen and if so is it acceptable? A quick test with
    // REVM wipes it too!
    let mut beneficiary_account = Account::from(beneficiary_account_info.clone());
    beneficiary_account.mark_touch();
    beneficiary_account
}
