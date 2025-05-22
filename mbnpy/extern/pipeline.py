from .dag import DAG

class Pipeline:
    """Create a pipeline by chaining multiple tasks and identifying dependencies."""

    def __init__(self):
        self.tasks = DAG()


    def task(self, depends_on=None):
        """Add new task to the pipeline and specify dependency task (optional)."""
        def inner(func):
            if depends_on:
                self.tasks.add(depends_on, func)
            else:
                self.tasks.add(func)
            return func
        return inner


    def run(self, *args):
        """Execute the pipeline and return each task results."""
        sorted_tasks = self.tasks.sort()
        completed = {}

        for task in sorted_tasks:
            for depend_on, nodes in self.tasks.graph.items():
                if task in nodes:
                    completed[task] = task(completed[depend_on])
            if task not in completed:
                if sorted_tasks.index(task) == 0:
                    completed[task] = task(*args)
                else:
                    completed[task] = task()

        return completed
