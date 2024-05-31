use std::{
    cmp::{max, min},
    num::NonZeroUsize,
    sync::{
        // TODO: Fine-tune all atomic `Ordering`s.
        // We're starting with `Relaxed` for maximum performance
        // without any issue so far. When in trouble, we can
        // retry `SeqCst` for robustness.
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
        Mutex,
    },
};

use ahash::{AHashMap, AHashSet};
use alloy_primitives::{Address, U256};
use revm::primitives::{TransactTo, TxEnv};

use crate::{ExecutionTask, Task, TxIdx, TxIncarnationStatus, TxVersion, ValidationTask};

// The BlockSTM collaborative scheduler coordinates execution & validation
// tasks among threads.
//
// To pick a task, threads increment the smaller of the (execution and
// validation) task counters until they find a task that is ready to be
// performed. To redo a task for a transaction, the thread updates the status
// and reduces the corresponding counter to the transaction index if it had a
// larger value.
//
// An incarnation may write to a memory location that was previously
// read by a higher transaction. Thus, when an incarnation finishes, new
// validation tasks are created for higher transactions.
//
// Validation tasks are scheduled optimistically and in parallel. Identifying
// validation failures and aborting incarnations as soon as possible is critical
// for performance, as any incarnation that reads values written by an
// incarnation that aborts also must abort.
// When an incarnation writes only to a subset of memory locations written
// by the previously completed incarnation of the same transaction, we schedule
// validation just for the incarnation. This is sufficient as the whole write
// set of the previous incarnation is marked as ESTIMATE during the abort.
// The abort leads to optimistically creating validation tasks for higher
// transactions. Threads that perform these tasks can already detect validation
// failure due to the ESTIMATE markers on memory locations, instead of waiting
// for a subsequent incarnation to finish.
#[derive(Debug, Default)]
pub(crate) struct Scheduler {
    /// The number of transactions in this block.
    block_size: usize,
    /// The first transaction going parallel that needs validation.
    starting_validation_idx: usize,
    /// The next transaction to try and execute.
    execution_idx: AtomicUsize,
    /// The next tranasction to try and validate.
    validation_idx: AtomicUsize,
    /// Number of ongoing execution and validation tasks.
    num_active_tasks: AtomicUsize,
    /// Number of times a task index was decreased.
    decrease_cnt: AtomicUsize,
    /// Marker for completion
    done_marker: AtomicBool,
    /// The most up-to-date incarnation number (initially 0) and
    /// the status of this incarnation.
    transactions_status: Vec<Mutex<TxIncarnationStatus>>,
    /// The list of dependent transactions to resumne when the
    /// key transaction is re-executed.
    transactions_dependents: Vec<Mutex<AHashSet<TxIdx>>>,
    /// A list of optional dependencies flagged during preprocessing.
    /// For instance, for a transaction to depend on two lower others,
    /// one send to the same recipient address, and one is from
    /// the same sender. We cannot casually check if all dependencies
    /// are clear with the dependents map as it can only lock the
    /// dependency. Two dependencies may check at the same time
    /// before they clear and think that the dependent is not yet
    /// ready, making it forever unexecuted.
    // TODO: Build a fuller dependency graph.
    transactions_dependencies: AHashMap<TxIdx, Mutex<AHashSet<TxIdx>>>,
    // We use `Vec` for dependents to simplify runtime update code.
    // We use `HashMap` for dependencies as we're only adding
    // them during preprocessing and removing them during processing.
    // The undelrying `HashSet` is to simplify index deduplication logic
    // while adding new dependencies.
    // TODO: Intuitively both should share a smiliar data structure?
}

impl Scheduler {
    // Prepare for a new execution run. Try to reuse previously allocated memory
    // as much as possible, instead of deallocating then allocating anew.
    // TODO: Make this as fast as possible.
    // For instance, to operate on an intermediate container instead of updating the
    // dependencies mutex as we go.
    pub(crate) fn prepare(&mut self, beneficiary_address: &Address, txs: &[TxEnv]) -> NonZeroUsize {
        // Marking transactions across same sender & recipient as dependents as they
        // cross-depend at the `AccountInfo` level (reading & writing to nonce & balance).
        // This is critical to avoid this nasty race condition:
        // 1. A spends money
        // 2. B sends money to A
        // 3. A spends money
        // Without (3) depending on (2), (2) may race and write to A first, then (1) comes
        // second flagging (2) for re-execution and execute (3) as dependency. (3) would
        // panic with a nonce error reading from (2) before it rewrites the new nonce
        // reading from (1).
        let mut tx_idxes_by_address: AHashMap<Address, Vec<TxIdx>> = AHashMap::new();
        let mut transactions_dependents: AHashMap<TxIdx, AHashSet<TxIdx>> = AHashMap::new();
        self.transactions_dependencies.clear();
        for (tx_idx, tx) in txs.iter().enumerate() {
            // We check for a non-empty value that guarantees to update the balance of the
            // recipient, to avoid smart contract interactions that only some storage.
            let mut recipient_with_changed_balance = None;
            if let TransactTo::Call(to) = tx.transact_to {
                if tx.value != U256::ZERO {
                    recipient_with_changed_balance = Some(to);
                }
            }

            // The first transaction never has dependencies.
            if tx_idx > 0 {
                // Register a lower transaction as this one's dependency.
                let mut register_dependency = |dependency_idx: usize| {
                    transactions_dependents
                        .entry(dependency_idx)
                        .or_default()
                        .insert(tx_idx);
                    self.transactions_dependencies
                        .entry(tx_idx)
                        .or_default()
                        .get_mut()
                        .unwrap()
                        .insert(dependency_idx);
                };

                if &tx.caller == beneficiary_address
                    || recipient_with_changed_balance.is_some_and(|to| &to == beneficiary_address)
                {
                    register_dependency(tx_idx - 1);
                } else {
                    if let Some(prev_idx) = tx_idxes_by_address
                        .get(&tx.caller)
                        .and_then(|tx_idxs| tx_idxs.last())
                    {
                        register_dependency(*prev_idx);
                    }
                    if let Some(to) = recipient_with_changed_balance {
                        if let Some(prev_idx) = tx_idxes_by_address
                            .get(&to)
                            .and_then(|tx_idxs| tx_idxs.last())
                        {
                            register_dependency(*prev_idx);
                        }
                    }
                }
            }

            // Register this transaction to the sender & recipient index maps.
            tx_idxes_by_address
                .entry(tx.caller)
                .or_default()
                .push(tx_idx);
            if let Some(to) = recipient_with_changed_balance {
                tx_idxes_by_address.entry(to).or_default().push(tx_idx);
            }
        }

        // Don't bother to evaluate the first fully sequential chain
        self.starting_validation_idx = 1;
        while transactions_dependents
            .get(&(self.starting_validation_idx - 1))
            .is_some_and(|deps| deps.contains(&self.starting_validation_idx))
        {
            self.starting_validation_idx += 1;
        }

        // Reset atomics
        self.execution_idx.store(0, Relaxed);
        self.validation_idx
            .store(self.starting_validation_idx, Relaxed);
        self.decrease_cnt.store(0, Relaxed);
        self.num_active_tasks.store(0, Relaxed);
        self.done_marker.store(false, Relaxed);

        // Reset transactions' status & dependents
        self.block_size = txs.len();
        for i in 0..self.block_size {
            let dependents = transactions_dependents.remove(&i).unwrap_or_default();
            let status = if self.transactions_dependencies.contains_key(&i) {
                TxIncarnationStatus::Aborting(0)
            } else {
                TxIncarnationStatus::ReadyToExecute(0)
            };
            // TODO: Assert the status & dependents lists are of equal length?
            if i < self.transactions_status.len() {
                *index_mutex!(self.transactions_status, i) = status;
                *index_mutex!(self.transactions_dependents, i) = dependents;
            } else {
                self.transactions_status.push(Mutex::new(status));
                self.transactions_dependents.push(Mutex::new(dependents));
            }
        }

        let min_concurrency_level = NonZeroUsize::new(2).unwrap();
        // This division by 2 means a thread must complete ~4 tasks to justify
        // its overheads.
        // TODO: Fine tune for edge cases given the dependency data above.
        NonZeroUsize::new(self.block_size / 2)
            .unwrap_or(min_concurrency_level)
            .max(min_concurrency_level)
    }

    pub(crate) fn done(&self) -> bool {
        self.done_marker.load(Relaxed)
    }

    fn decrease_execution_idx(&self, target_idx: TxIdx) {
        if self.execution_idx.fetch_min(target_idx, Relaxed) > target_idx {
            self.decrease_cnt.fetch_add(1, Relaxed);
        }
    }

    fn decrease_validation_idx(&self, target_idx: TxIdx) {
        if self.validation_idx.fetch_min(target_idx, Relaxed) > target_idx {
            self.decrease_cnt.fetch_add(1, Relaxed);
        }
    }

    fn check_done(&self) {
        let observed_cnt = self.decrease_cnt.load(Relaxed);
        let execution_idx = self.execution_idx.load(Relaxed);
        let validation_idx = self.validation_idx.load(Relaxed);
        let num_active_tasks = self.num_active_tasks.load(Relaxed);
        if min(execution_idx, validation_idx) >= self.block_size
            && num_active_tasks == 0
            && observed_cnt == self.decrease_cnt.load(Relaxed)
        {
            self.done_marker.store(true, Relaxed);
        }
    }

    fn try_incarnate(&self, mut tx_idx: TxIdx) -> Option<TxVersion> {
        while tx_idx < self.block_size {
            let mut transaction_status = index_mutex!(self.transactions_status, tx_idx);
            if let TxIncarnationStatus::ReadyToExecute(i) = *transaction_status {
                *transaction_status = TxIncarnationStatus::Executing(i);
                return Some(TxVersion {
                    tx_idx,
                    tx_incarnation: i,
                });
            }
            drop(transaction_status);
            tx_idx = self.execution_idx.fetch_add(1, Relaxed);
        }
        self.num_active_tasks.fetch_sub(1, Relaxed);
        None
    }

    fn next_version_to_execute(&self) -> Option<TxVersion> {
        if self.execution_idx.load(Relaxed) >= self.block_size {
            self.check_done();
            None
        } else {
            self.num_active_tasks.fetch_add(1, Relaxed);
            self.try_incarnate(self.execution_idx.fetch_add(1, Relaxed))
        }
    }

    fn next_version_to_validate(&self) -> Option<TxVersion> {
        if self.validation_idx.load(Relaxed) >= self.block_size {
            self.check_done();
            return None;
        }
        self.num_active_tasks.fetch_add(1, Relaxed);
        let mut validation_idx = self.validation_idx.fetch_add(1, Relaxed);
        while validation_idx < self.block_size {
            let transaction_status = index_mutex!(self.transactions_status, validation_idx);
            if let TxIncarnationStatus::Executed(i) = *transaction_status {
                return Some(TxVersion {
                    tx_idx: validation_idx,
                    tx_incarnation: i,
                });
            }
            drop(transaction_status);
            validation_idx = self.validation_idx.fetch_add(1, Relaxed);
        }
        self.num_active_tasks.fetch_sub(1, Relaxed);
        None
    }

    pub(crate) fn next_task(&self) -> Option<Task> {
        if self.validation_idx.load(Relaxed) < self.execution_idx.load(Relaxed) {
            self.next_version_to_validate().map(Task::Validation)
        } else {
            self.next_version_to_execute().map(Task::Execution)
        }
    }

    // Add `tx_idx` as a dependent of `blocking_tx_idx` so `tx_idx` is
    // re-executed when the next `blocking_tx_idx` incarnation is executed.
    // Return `false` if we encouter a race condition when `blocking_tx_idx`
    // gets re-executed before the dependency can be added.
    // TODO: Better error handling, including asserting that both indices are in range.
    pub(crate) fn add_dependency(&self, tx_idx: TxIdx, blocking_tx_idx: TxIdx) -> bool {
        // This is an important lock to prevent a race condition where the blocking
        // transaction completes re-execution before this dependecy can be added.
        let blocking_transaction_status = index_mutex!(self.transactions_status, blocking_tx_idx);
        if let TxIncarnationStatus::Executed(_) = *blocking_transaction_status {
            return false;
        }

        let mut transaction_status = index_mutex!(self.transactions_status, tx_idx);
        if let TxIncarnationStatus::Executing(i) = *transaction_status {
            *transaction_status = TxIncarnationStatus::Aborting(i);
            drop(transaction_status);

            // TODO: Better error handling here
            let mut blocking_dependents =
                index_mutex!(self.transactions_dependents, blocking_tx_idx);
            blocking_dependents.insert(tx_idx);
            drop(blocking_dependents);

            self.num_active_tasks.fetch_sub(1, Relaxed);
            return true;
        }

        unreachable!("Trying to abort & add dependency in non-executing state!")
    }

    // Be careful as this one is usually called as a sub-routine that is very
    // easy to dead-lock.
    fn set_ready_status(&self, tx_idx: TxIdx) {
        // TODO: Better error handling
        let mut transaction_status = index_mutex!(self.transactions_status, tx_idx);
        if let TxIncarnationStatus::Aborting(i) = *transaction_status {
            *transaction_status = TxIncarnationStatus::ReadyToExecute(i + 1)
        } else {
            unreachable!("Trying to resume in non-aborting state!")
        }
    }

    // Finish execution and resume dependents of a transaction incarnation.
    // When a new location was written, schedule the re-execution of all
    // higher transactions. If not, return the validation task to validate
    // only this incarnation. Return no task if we've already rolled back to
    // re-validating smaller transactions.
    pub(crate) fn finish_execution(
        &self,
        tx_version: TxVersion,
        wrote_new_location: bool,
    ) -> Option<ValidationTask> {
        // TODO: Better error handling
        let mut transaction_status = index_mutex!(self.transactions_status, tx_version.tx_idx);
        if let TxIncarnationStatus::Executing(i) = *transaction_status {
            // TODO: Assert that `i` equals `tx_version.tx_incarnation`?
            *transaction_status = TxIncarnationStatus::Executed(i);
            drop(transaction_status);

            // Resume dependent transactions
            let mut dependents = index_mutex!(self.transactions_dependents, tx_version.tx_idx);
            let mut min_dependent_idx = None;
            for tx_idx in dependents.iter() {
                if let Some(deps) = self.transactions_dependencies.get(tx_idx) {
                    // TODO: Better error handling
                    let mut deps = deps.lock().unwrap();
                    deps.retain(|dep_idx| dep_idx != &tx_version.tx_idx);
                    // Skip this dependent as it has other pending dependencies.
                    // Let the last one evoke it.
                    if !deps.is_empty() {
                        continue;
                    }
                }
                self.set_ready_status(*tx_idx);
                min_dependent_idx = match min_dependent_idx {
                    None => Some(*tx_idx),
                    Some(min_index) => Some(min(*tx_idx, min_index)),
                }
            }
            dependents.clear();
            drop(dependents);

            if let Some(min_idx) = min_dependent_idx {
                self.decrease_execution_idx(min_idx);
            }
            if self.validation_idx.load(Relaxed) > tx_version.tx_idx {
                // This incarnation wrote to a new location, so we must
                // re-evaluate it (via the immediately returned task returned
                // immediately) and all higher transactions in case they read
                // the new location.
                if wrote_new_location {
                    self.decrease_validation_idx(max(
                        tx_version.tx_idx + 1,
                        self.starting_validation_idx,
                    ));
                }
                if tx_version.tx_idx >= self.starting_validation_idx {
                    return Some(tx_version);
                }
            }
        } else {
            // TODO: Better error handling
            unreachable!("Trying to finish execution in a non-executing state")
        }
        self.num_active_tasks.fetch_sub(1, Relaxed);
        None
    }

    // Return whether the abort was successful. A successful abort leads to
    // scheduling the transaction for re-execution and the higher transactions
    // for validation during `finish_validation`. The scheduler ensures that only
    // one failing validation per version can lead to a successful abort.
    pub(crate) fn try_validation_abort(&self, tx_version: &TxVersion) -> bool {
        // TODO: Better error handling
        let mut transaction_status = index_mutex!(self.transactions_status, tx_version.tx_idx);
        if let TxIncarnationStatus::Executed(i) = *transaction_status {
            *transaction_status = TxIncarnationStatus::Aborting(i);
            true
        } else {
            false
        }
    }

    // When there is a successful abort, schedule the transaction for re-execution
    // and the higher transactions for validation. The re-execution task is returned
    // for the aborted transaction.
    pub(crate) fn finish_validation(
        &self,
        tx_version: &TxVersion,
        aborted: bool,
    ) -> Option<ExecutionTask> {
        if aborted {
            self.set_ready_status(tx_version.tx_idx);
            self.decrease_validation_idx(tx_version.tx_idx + 1);
            if self.execution_idx.load(Relaxed) > tx_version.tx_idx {
                return self.try_incarnate(tx_version.tx_idx);
            }
        }
        self.num_active_tasks.fetch_sub(1, Relaxed);
        None
    }
}
