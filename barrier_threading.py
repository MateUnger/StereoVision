import threading
import time

# Number of worker threads
NUM_THREADS = 4

# Create a barrier that will block until all NUM_THREADS have called .wait()
barrier = threading.Barrier(NUM_THREADS)

def worker(thread_id):
    """
    Each thread will perform some work, then wait on the barrier
    until all threads reach the same point (a "clock cycle").
    """
    for cycle in range(3):  # Example: perform 3 cycles
        # Simulate some work here
        time.sleep(0.5)
        print(f"Thread {thread_id} completed work in cycle {cycle}")

        # Wait for other threads
        barrier.wait()

        # Optional: Code to execute immediately after the barrier point
        #           (before next cycle of work)
        if thread_id == 0:
            print("All threads reached the barrier. Next cycle.\n")

def main():
    # Create and start threads
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()
    print("All threads have finished execution.")

if __name__ == "__main__":
    main()