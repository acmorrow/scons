# MIT License
#
# Copyright The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Serial and Parallel classes to execute build tasks.

The Jobs class provides a higher level interface to start,
stop, and wait on jobs.
"""

import SCons.compat

import os
import signal
import threading

import SCons.Errors
import SCons.Warnings

# The default stack size (in kilobytes) of the threads used to execute
# jobs in parallel.
#
# We use a stack size of 256 kilobytes. The default on some platforms
# is too large and prevents us from creating enough threads to fully
# parallelized the build. For example, the default stack size on linux
# is 8 MBytes.

explicit_stack_size = None
default_stack_size = 256

interrupt_msg = 'Build interrupted.'


class InterruptState:
    def __init__(self):
        self.interrupted = False

    def set(self):
        self.interrupted = True

    def __call__(self):
        return self.interrupted


class Jobs:
    """An instance of this class initializes N jobs, and provides
    methods for starting, stopping, and waiting on all N jobs.
    """

    def __init__(self, num, taskmaster):
        """
        Create 'num' jobs using the given taskmaster.

        If 'num' is 1 or less, then a serial job will be used,
        otherwise a parallel job with 'num' worker threads will
        be used.

        The 'num_jobs' attribute will be set to the actual number of jobs
        allocated.  If more than one job is requested but the Parallel
        class can't do it, it gets reset to 1.  Wrapping interfaces that
        care should check the value of 'num_jobs' after initialization.
        """

        self.job = None
        if num > 1:
            stack_size = explicit_stack_size
            if stack_size is None:
                stack_size = default_stack_size

            try:
                self.job = Parallel(taskmaster, num, stack_size)
                self.num_jobs = num
            except NameError:
                pass
        if self.job is None:
            self.job = Serial(taskmaster)
            self.num_jobs = 1

    def run(self, postfunc=lambda: None):
        """Run the jobs.

        postfunc() will be invoked after the jobs has run. It will be
        invoked even if the jobs are interrupted by a keyboard
        interrupt (well, in fact by a signal such as either SIGINT,
        SIGTERM or SIGHUP). The execution of postfunc() is protected
        against keyboard interrupts and is guaranteed to run to
        completion."""
        self._setup_sig_handler()
        try:
            self.job.start()
        finally:
            postfunc()
            self._reset_sig_handler()

    def were_interrupted(self):
        """Returns whether the jobs were interrupted by a signal."""
        return self.job.interrupted()

    def _setup_sig_handler(self):
        """Setup an interrupt handler so that SCons can shutdown cleanly in
        various conditions:

          a) SIGINT: Keyboard interrupt
          b) SIGTERM: kill or system shutdown
          c) SIGHUP: Controlling shell exiting

        We handle all of these cases by stopping the taskmaster. It
        turns out that it's very difficult to stop the build process
        by throwing asynchronously an exception such as
        KeyboardInterrupt. For example, the python Condition
        variables (threading.Condition) and queues do not seem to be
        asynchronous-exception-safe. It would require adding a whole
        bunch of try/finally block and except KeyboardInterrupt all
        over the place.

        Note also that we have to be careful to handle the case when
        SCons forks before executing another process. In that case, we
        want the child to exit immediately.
        """
        def handler(signum, stack, self=self, parentpid=os.getpid()):
            if os.getpid() == parentpid:
                self.job.taskmaster.stop()
                self.job.interrupted.set()
            else:
                os._exit(2)  # pylint: disable=protected-access

        self.old_sigint  = signal.signal(signal.SIGINT, handler)
        self.old_sigterm = signal.signal(signal.SIGTERM, handler)
        try:
            self.old_sighup = signal.signal(signal.SIGHUP, handler)
        except AttributeError:
            pass
        if (self.old_sigint is None) or (self.old_sigterm is None) or \
                (hasattr(self, "old_sighup") and self.old_sighup is None):
            msg = "Overwritting previous signal handler which was not installed from Python. " + \
                "Will not be able to reinstate and so will return to default handler."
            SCons.Warnings.warn(SCons.Warnings.SConsWarning, msg)

    def _reset_sig_handler(self):
        """Restore the signal handlers to their previous state (before the
         call to _setup_sig_handler()."""
        sigint_to_use = self.old_sigint if self.old_sigint is not None else signal.SIG_DFL
        sigterm_to_use = self.old_sigterm if self.old_sigterm is not None else signal.SIG_DFL
        signal.signal(signal.SIGINT, sigint_to_use)
        signal.signal(signal.SIGTERM, sigterm_to_use)
        try:
            sigterm_to_use = self.old_sighup if self.old_sighup is not None else signal.SIG_DFL
            signal.signal(signal.SIGHUP, sigterm_to_use)
        except AttributeError:
            pass

class Serial:
    """This class is used to execute tasks in series, and is more efficient
    than Parallel, but is only appropriate for non-parallel builds. Only
    one instance of this class should be in existence at a time.

    This class is not thread safe.
    """

    def __init__(self, taskmaster):
        """Create a new serial job given a taskmaster.

        The taskmaster's next_task() method should return the next task
        that needs to be executed, or None if there are no more tasks. The
        taskmaster's executed() method will be called for each task when it
        is successfully executed, or failed() will be called if it failed to
        execute (e.g. execute() raised an exception)."""

        self.taskmaster = taskmaster
        self.interrupted = InterruptState()

    def start(self):
        """Start the job. This will begin pulling tasks from the taskmaster
        and executing them, and return when there are no more tasks. If a task
        fails to execute (i.e. execute() raises an exception), then the job will
        stop."""

        while True:
            task = self.taskmaster.next_task()

            if task is None:
                break

            try:
                task.prepare()
                if task.needs_execute():
                    task.execute()
            except Exception:
                if self.interrupted():
                    try:
                        raise SCons.Errors.BuildError(
                            task.targets[0], errstr=interrupt_msg)
                    except:
                        task.exception_set()
                else:
                    task.exception_set()

                # Let the failed() callback function arrange for the
                # build to stop if that's appropriate.
                task.failed()
            else:
                task.executed()

            task.postprocess()
        self.taskmaster.cleanup()


# Trap import failure so that everything in the Job module but the
# Parallel class (and its dependent classes) will work if the interpreter
# doesn't support threads.
try:
    import queue
    import threading
except ImportError:
    pass
else:
    class Worker(threading.Thread):
        """A worker thread waits on a task to be posted to its request queue,
        dequeues the task, executes it, and posts a tuple including the task
        and a boolean indicating whether the task executed successfully. """

        def __init__(self, requestQueue, resultsQueue, interrupted):
            super().__init__()
            self.daemon = True
            self.requestQueue = requestQueue
            self.resultsQueue = resultsQueue
            self.interrupted = interrupted
            self.start()

        def run(self):
            while True:
                task = self.requestQueue.get()

                if task is None:
                    # The "None" value is used as a sentinel by
                    # ThreadPool.cleanup().  This indicates that there
                    # are no more tasks, so we should quit.
                    break

                try:
                    if self.interrupted():
                        raise SCons.Errors.BuildError(
                            task.targets[0], errstr=interrupt_msg)
                    task.execute()
                except:
                    task.exception_set()
                    ok = False
                else:
                    ok = True

                self.resultsQueue.put((task, ok))

    class ThreadPool:
        """This class is responsible for spawning and managing worker threads."""

        def __init__(self, num, stack_size, interrupted):
            """Create the request and reply queues, and 'num' worker threads.

            One must specify the stack size of the worker threads. The
            stack size is specified in kilobytes.
            """
            self.requestQueue = queue.Queue(0)
            self.resultsQueue = queue.Queue(0)

            try:
                prev_size = threading.stack_size(stack_size*1024)
            except AttributeError as e:
                # Only print a warning if the stack size has been
                # explicitly set.
                if explicit_stack_size is not None:
                    msg = "Setting stack size is unsupported by this version of Python:\n    " + \
                        e.args[0]
                    SCons.Warnings.warn(SCons.Warnings.StackSizeWarning, msg)
            except ValueError as e:
                msg = "Setting stack size failed:\n    " + str(e)
                SCons.Warnings.warn(SCons.Warnings.StackSizeWarning, msg)

            # Create worker threads
            self.workers = []
            for _ in range(num):
                worker = Worker(self.requestQueue, self.resultsQueue, interrupted)
                self.workers.append(worker)

            if 'prev_size' in locals():
                threading.stack_size(prev_size)

        def put(self, task):
            """Put task into request queue."""
            self.requestQueue.put(task)

        def get(self):
            """Remove and return a result tuple from the results queue."""
            return self.resultsQueue.get()

        def preparation_failed(self, task):
            self.resultsQueue.put((task, False))

        def cleanup(self):
            """
            Shuts down the thread pool, giving each worker thread a
            chance to shut down gracefully.
            """
            # For each worker thread, put a sentinel "None" value
            # on the requestQueue (indicating that there's no work
            # to be done) so that each worker thread will get one and
            # terminate gracefully.
            for _ in self.workers:
                self.requestQueue.put(None)

            # Wait for all of the workers to terminate.
            #
            # If we don't do this, later Python versions (2.4, 2.5) often
            # seem to raise exceptions during shutdown.  This happens
            # in requestQueue.get(), as an assertion failure that
            # requestQueue.not_full is notified while not acquired,
            # seemingly because the main thread has shut down (or is
            # in the process of doing so) while the workers are still
            # trying to pull sentinels off the requestQueue.
            #
            # Normally these terminations should happen fairly quickly,
            # but we'll stick a one-second timeout on here just in case
            # someone gets hung.
            for worker in self.workers:
                worker.join(1.0)
            self.workers = []

    class LegacyParallel:
        """This class is used to execute tasks in parallel, and is somewhat
        less efficient than Serial, but is appropriate for parallel builds.

        This class is thread safe.
        """

        def __init__(self, taskmaster, num, stack_size):
            """Create a new parallel job given a taskmaster.

            The taskmaster's next_task() method should return the next
            task that needs to be executed, or None if there are no more
            tasks. The taskmaster's executed() method will be called
            for each task when it is successfully executed, or failed()
            will be called if the task failed to execute (i.e. execute()
            raised an exception).

            Note: calls to taskmaster are serialized, but calls to
            execute() on distinct tasks are not serialized, because
            that is the whole point of parallel jobs: they can execute
            multiple tasks simultaneously. """

            self.taskmaster = taskmaster
            self.interrupted = InterruptState()
            self.tp = ThreadPool(num, stack_size, self.interrupted)

            self.maxjobs = num

        def start(self):
            """Start the job. This will begin pulling tasks from the
            taskmaster and executing them, and return when there are no
            more tasks. If a task fails to execute (i.e. execute() raises
            an exception), then the job will stop."""

            jobs = 0

            while True:
                # Start up as many available tasks as we're
                # allowed to.
                while jobs < self.maxjobs:
                    task = self.taskmaster.next_task()
                    if task is None:
                        break

                    try:
                        # prepare task for execution
                        task.prepare()
                    except:
                        task.exception_set()
                        task.failed()
                        task.postprocess()
                    else:
                        if task.needs_execute():
                            # dispatch task
                            self.tp.put(task)
                            jobs += 1
                        else:
                            task.executed()
                            task.postprocess()

                if not task and not jobs: break

                # Let any/all completed tasks finish up before we go
                # back and put the next batch of tasks on the queue.
                while True:
                    task, ok = self.tp.get()
                    jobs -= 1

                    if ok:
                        task.executed()
                    else:
                        if self.interrupted():
                            try:
                                raise SCons.Errors.BuildError(
                                    task.targets[0], errstr=interrupt_msg)
                            except:
                                task.exception_set()

                        # Let the failed() callback function arrange
                        # for the build to stop if that's appropriate.
                        task.failed()

                    task.postprocess()

                    if self.tp.resultsQueue.empty():
                        break

            self.tp.cleanup()
            self.taskmaster.cleanup()


    class NewParallel:

        class Worker(threading.Thread):
            def __init__(self, owner):
                super().__init__()
                self.daemon = True
                self.owner = owner
                self.start()

            def run(self):
                self.owner._work()

        def __init__(self, taskmaster, num, stack_size):
            self.taskmaster = taskmaster
            self.num_workers = num
            self.stack_size = stack_size
            self.interrupted = InterruptState()
            self.workers = []

            # The `tm_lock` is what ensures that we only have one thread
            # interacting with the taskmaster at a time. It also protects
            # access to our state that gets updated concurrently. All condition
            # variables are associated with this lock.
            self.tm_lock = threading.Lock()

            # The `searching` state and the `can_search_cv` are used
            # to manage a leader / follower pattern for access to the
            # taskmaster.
            self.searching = False
            self.can_search_cv = threading.Condition(self.tm_lock)

            # The following state helps us manage the state machine as
            # we decide whether there is more work to do or whether we
            # need to stall on job completions.
            self.jobs = 0
            self.completed = False
            self.stalled = False

            # self.idle_epoch = 0
            # self.idlers = 0
            # self.idlers_cv = threading.Condition(self.tm_lock)
            

            # The queue of tasks that have completed execution. The
            # next thread to obtain the tm_lock will retire all of
            # them.
            self.results_queue_lock = threading.Lock()
            self.results_queue = []

        def start(self):
            self._start_workers()
            for worker in self.workers:
                worker.join()
            self.workers = []
            self.taskmaster.cleanup()

        def _start_workers(self):
            prev_size = self._adjust_stack_size()
            for _ in range(self.num_workers):
                self.workers.append(NewParallel.Worker(self))
            self._restore_stack_size(prev_size)

        def _adjust_stack_size(self):
            try:
                prev_size = threading.stack_size(self.stack_size*1024)
                return prev_size
            except AttributeError as e:
                # Only print a warning if the stack size has been
                # explicitly set.
                if explicit_stack_size is not None:
                    msg = "Setting stack size is unsupported by this version of Python:\n    " + \
                        e.args[0]
                    SCons.Warnings.warn(SCons.Warnings.StackSizeWarning, msg)
            except ValueError as e:
                msg = "Setting stack size failed:\n    " + str(e)
                SCons.Warnings.warn(SCons.Warnings.StackSizeWarning, msg)

            return None

        def _restore_stack_size(self, prev_size):
            if prev_size is not None:
                threading.stack_size(prev_size)

        def _work(self):
            while True:

                # Obtain the `tm_lock` mutex that gives us exclusive
                # access to the taskmaster.
                with self.can_search_cv:

                    if self.stalled:
                        # print(f"XXX {threading.get_ident()} Detected stall at search start")
                        with self.results_queue_lock:
                            if self.results_queue:
                                self.stalled = False
                                self.searching = False
                                # print(f"XXX {threading.get_ident()} This thread has new results, dropping stall mark")

                    # Assuming we haven't been marked completed, wait
                    # until there is no other thread searching.
                    while not self.completed and self.searching:
                        # print(f"XXX {threading.get_ident()} Waiting to search")
                        self.can_search_cv.wait()

                    # If someone set the completed flag, bail.
                    if self.completed:
                        break

                    # Set the searching flag to indicate that a thread
                    # is currently in the critical section for
                    # taskmaster work.
                    self.searching = True

                    # print(f"XXX {threading.get_ident()} Thread is searching")

                    # Bulk acquire the tasks in the results queue
                    # under the result queue lock, then process them
                    # all outside the lock. We need to process the
                    # tasks in the results queue before looking for
                    # new work because we might be unable to find new
                    # work if we don't.
                    results_queue = []
                    with self.results_queue_lock:
                        results_queue, self.results_queue = self.results_queue, results_queue

                    for (rtask, rresult) in results_queue:
                        if rresult:
                            rtask.executed()
                        else:
                            if self.interrupted():
                                try:
                                    raise SCons.Errors.BuildError(
                                        rtask.targets[0], errstr=interrupt_msg)
                                except:
                                    rtask.exception_set()

                            # Let the failed() callback function arrange
                            # for the build to stop if that's appropriate.
                            rtask.failed()

                        rtask.postprocess()
                        self.jobs -= 1

                    # If we had results to process, and there are idle
                    # threads, awaken them all. We need to awaken them
                    # all because processing even one result might
                    # unblock an arbitrary amount of new work.
                    if results_queue and self.stalled:
                        # self.stalled = False
                        pass # XXX {threading.get_ident()} what should happen here. Presumably a notify?
                        #self.idle_epoch += 1
                        #self.idlers_cv.notify_all()

                    # We are done with any task objects that
                    # were in the results queue.
                    results_queue.clear()

                    # Now, turn the crank on the taskmaster until we
                    # either run out of tasks, or find a task that
                    # needs execution. If we run out of tasks, go idle
                    # until results arrive if jobs are pending, or
                    # mark the walk as complete if not.
                    while self.searching and not self.stalled:
                        task = self.taskmaster.next_task()

                        if task:
                            # We found a task. Walk it through the
                            # task lifecycle. If it does not need
                            # execution, just complete the task and
                            # look for the next one. Otherwise,
                            # indicate that we are no longer searching
                            # so we can drop out of this loop, execute
                            # the task outside the lock, and allow
                            # another thread in to search.
                            try:
                                task.prepare()
                            except:
                                task.exception_set()
                                task.failed()
                                task.postprocess()
                            else:
                                if not task.needs_execute():
                                    task.executed()
                                    task.postprocess()
                                else:
                                    # print(f"XXX {threading.get_ident()} Thread found work to requiring execution")
                                    self.jobs += 1
                                    self.searching = False
                                    self.can_search_cv.notify()

                        else:
                            # We failed to find a task, so this thread
                            # cannot continue turning the taskmaster
                            # crank. Note that we are no longer
                            # searching so that we drop out of this
                            # loop without `task` set.
                            if self.jobs:
                                # No task was found, but there are
                                # outstanding jobs executing that
                                # might unblock new tasks when they
                                # complete. We are stalled.
                                self.stalled = True
                                # print(f"XXX {threading.get_ident()} Stalled by failed search")
                            else:
                                self.searching = False
                                # We didn't find a task and there are
                                # no jobs outstanding, so there is
                                # nothing that will ever return
                                # results which might unblock new
                                # tasks. We can conclude that the walk
                                # is complete. If we are the first to
                                # notice, set the completed flag and
                                # awaken anyone sleeping on the
                                # condvar. Also advance the idle epoch
                                # and awake all idle threads so they
                                # can terminate.
                                # print(f"XXX {threading.get_ident()} Failed search with no jobs")
                                if not self.completed:
                                    # print(f"XXX {threading.get_ident()} Noting completion")
                                    self.completed = True
                                    self.can_search_cv.notify_all()

                # We no longer hold `tm_lock` here. If we have a task,
                # we can now execute it. If there are threads waiting
                # to search, one of them can now begin turning the
                # taskmaster crank in parallel.
                if task:
                    ok = True
                    try:
                        if self.interrupted():
                            raise SCons.Errors.BuildError(
                                task.targets[0], errstr=interrupt_msg)
                        task.execute()
                    except:
                        ok = False
                        task.exception_set()

                    # Grab the results lock and enqueue the executed task
                    # and state. The next thread into the searching loop
                    # will complete the postprocessing work under the
                    # taskmaster lock.
                    with self.results_queue_lock:
                        self.results_queue.append((task, ok))

                    # We have no further interest in `task`.
                    task = None

    Parallel = NewParallel

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
