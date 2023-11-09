import time

class Monitor():
    def __init__(self):
        # Initialize instance variables to keep track of metrics
        self.total_execution_time = 0.0
        self.total_tokens_generated = 0
        self.num_calls = 0
        self.token_by_second = 0

    def update_metrics(self, start_time, nbr_tokens):
        end_time = time.time()  # Measure the end time
        execution_time = end_time - start_time

        # Update cumulative metrics
        self.total_execution_time += execution_time
        self.total_tokens_generated += nbr_tokens
        self.token_by_second = nbr_tokens / execution_time
        self.average_token_by_second = self.total_tokens_generated / self.total_execution_time
        self.num_calls += 1

    def metrics(self):
        return {
            "total_exec_time" : self.total_execution_time ,
            "total_token_generated" : self.total_tokens_generated,
            "number_call": self.num_calls,
            "token_by_second": self.token_by_second,
            "average_token_second": self.average_token_by_second
            } 

    # Method to display current metrics
    def display_metrics(self):
        print(f"Total Execution Time: {self.total_execution_time} seconds")
        print(f"Total Tokens Generated: {self.total_tokens_generated}")
        print(f"Number of Calls: {self.num_calls}")
        print(f"Tokens per second: {self.token_by_second:.2f}")
        print(f"Average Tokens per second: {self.average_token_by_second:.2f}")        

# Example usage
monitor = Monitor()
start_time = time.time()
j = 0
for i in range(0, 10000):
    j+=1
monitor.update_metrics(start_time,100)
#monitor.display_metrics()
print(monitor.metrics())
