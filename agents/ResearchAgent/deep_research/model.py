"""
Deep Research Plan Model

This module defines the data structures for managing deep research plans,
including task dependencies, execution order, and context management.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class ResearchTask:
    """Represents a single research task in the plan."""
    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate task data after initialization."""
        if not self.id:
            raise ValueError("Task ID cannot be empty")
        if not self.description:
            raise ValueError("Task description cannot be empty")
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class ContextSummary:
    """Represents a summary of completed tasks for context passing."""
    task_id: str
    summary: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "summary": self.summary,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextSummary':
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            summary=data["summary"],
            created_at=datetime.fromisoformat(data["created_at"])
        )


class ResearchPlan:
    """
    Manages a deep research plan with task dependencies and execution order.
    
    This class handles:
    - Parsing JSON plan data from the planner
    - Managing task dependencies and execution order
    - Tracking task status and results
    - Managing context summaries for task execution
    """
    
    def __init__(self, plan_data: List[Dict[str, Any]], plan_id: Optional[str] = None):
        """
        Initialize a research plan from JSON data.
        
        Args:
            plan_data: List of task dictionaries from the planner
            plan_id: Optional plan identifier (generated if not provided)
        """
        self.plan_id = plan_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.tasks: Dict[str, ResearchTask] = {}
        self.context_summaries: List[ContextSummary] = []
        self.execution_order: List[str] = []
        
        # Parse and validate plan data
        self._parse_plan_data(plan_data)
        self._calculate_execution_order()
        
    def _parse_plan_data(self, plan_data: List[Dict[str, Any]]) -> None:
        """Parse JSON plan data into ResearchTask objects."""
        for task_data in plan_data:
            task = ResearchTask(
                id=task_data["id"],
                description=task_data["description"],
                depends_on=task_data.get("depends_on", [])
            )
            self.tasks[task.id] = task
    
    def _calculate_execution_order(self) -> None:
        """
        Calculate the execution order based on task dependencies using topological sort.
        
        This ensures that tasks are executed in the correct order, with dependencies
        resolved before dependent tasks are started.
        """
        # Build dependency graph
        in_degree = {task_id: 0 for task_id in self.tasks}
        graph = {task_id: [] for task_id in self.tasks}
        
        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep in self.tasks:
                    graph[dep].append(task.id)
                    in_degree[task.id] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current_task = queue.pop(0)
            execution_order.append(current_task)
            
            # Reduce in-degree for dependent tasks
            for dependent in graph[current_task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(execution_order) != len(self.tasks):
            raise ValueError("Circular dependency detected in research plan")
        
        self.execution_order = execution_order
    
    def get_next_executable_tasks(self) -> List[ResearchTask]:
        """
        Get tasks that can be executed next (dependencies satisfied and not completed).
        
        Returns:
            List of tasks ready for execution
        """
        executable_tasks = []
        
        for task_id in self.execution_order:
            task = self.tasks[task_id]
            
            # Skip if already completed or failed
            if task.status in ["completed", "failed"]:
                continue
            
            # Check if all dependencies are completed
            dependencies_satisfied = all(
                self.tasks[dep_id].status == "completed"
                for dep_id in task.depends_on
                if dep_id in self.tasks
            )
            
            if dependencies_satisfied and task.status == "pending":
                executable_tasks.append(task)
        
        return executable_tasks
    
    def get_concurrent_tasks(self) -> List[List[ResearchTask]]:
        """
        Get tasks grouped by execution level (tasks that can run concurrently).
        
        Returns:
            List of task groups, where each group can be executed concurrently
        """
        concurrent_groups = []
        remaining_tasks = set(self.tasks.keys())
        
        while remaining_tasks:
            # Find tasks with no pending dependencies
            current_level = []
            for task_id in list(remaining_tasks):
                task = self.tasks[task_id]
                
                # Check if all dependencies are completed or not in remaining tasks
                dependencies_satisfied = all(
                    dep_id not in remaining_tasks or self.tasks[dep_id].status == "completed"
                    for dep_id in task.depends_on
                )
                
                if dependencies_satisfied and task.status == "pending":
                    current_level.append(task)
                    remaining_tasks.remove(task_id)
            
            if not current_level:
                # No more tasks can be executed (circular dependency or all completed)
                break
            
            concurrent_groups.append(current_level)
        
        return concurrent_groups
    
    def mark_task_in_progress(self, task_id: str) -> bool:
        """Mark a task as in progress."""
        if task_id in self.tasks and self.tasks[task_id].status == "pending":
            self.tasks[task_id].status = "in_progress"
            return True
        return False
    
    def mark_task_completed(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed with results."""
        if task_id in self.tasks and self.tasks[task_id].status == "in_progress":
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result = result
            self.tasks[task_id].completed_at = datetime.utcnow()
            return True
        return False
    
    def mark_task_failed(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].result = {"error": error}
            self.tasks[task_id].completed_at = datetime.utcnow()
            return True
        return False
    
    def add_context_summary(self, task_id: str, summary: str) -> None:
        """Add a context summary for a completed task."""
        context_summary = ContextSummary(task_id=task_id, summary=summary)
        self.context_summaries.append(context_summary)
    
    def get_context_summary_text(self) -> str:
        """
        Get all context summaries as a single text for passing to next tasks.
        
        Returns:
            Concatenated context summaries
        """
        if not self.context_summaries:
            return ""
        
        summaries = []
        for context in self.context_summaries:
            summaries.append(f"Task {context.task_id}: {context.summary}")
        
        return "\n\n".join(summaries)
    
    def get_task_by_id(self, task_id: str) -> Optional[ResearchTask]:
        """Get a task by its ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ResearchTask]:
        """Get all tasks in execution order."""
        return [self.tasks[task_id] for task_id in self.execution_order]
    
    def get_completed_tasks(self) -> List[ResearchTask]:
        """Get all completed tasks."""
        return [task for task in self.tasks.values() if task.status == "completed"]
    
    def get_pending_tasks(self) -> List[ResearchTask]:
        """Get all pending tasks."""
        return [task for task in self.tasks.values() if task.status == "pending"]
    
    def get_failed_tasks(self) -> List[ResearchTask]:
        """Get all failed tasks."""
        return [task for task in self.tasks.values() if task.status == "failed"]
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == "completed" for task in self.tasks.values())
    
    def get_progress(self) -> Dict[str, Any]:
        """Get plan progress statistics."""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.get_completed_tasks())
        failed_tasks = len(self.get_failed_tasks())
        pending_tasks = len(self.get_pending_tasks())
        
        return {
            "plan_id": self.plan_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": pending_tasks,
            "completion_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "created_at": self.created_at.isoformat(),
            "context_summaries_count": len(self.context_summaries)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at.isoformat(),
            "tasks": {
                task_id: {
                    "id": task.id,
                    "description": task.description,
                    "depends_on": task.depends_on,
                    "status": task.status,
                    "result": task.result,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                for task_id, task in self.tasks.items()
            },
            "execution_order": self.execution_order,
            "context_summaries": [ctx.to_dict() for ctx in self.context_summaries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchPlan':
        """Create plan from dictionary."""
        plan = cls.__new__(cls)
        plan.plan_id = data["plan_id"]
        plan.created_at = datetime.fromisoformat(data["created_at"])
        plan.execution_order = data["execution_order"]
        
        # Reconstruct tasks
        plan.tasks = {}
        for task_id, task_data in data["tasks"].items():
            task = ResearchTask(
                id=task_data["id"],
                description=task_data["description"],
                depends_on=task_data["depends_on"],
                status=task_data["status"],
                result=task_data["result"],
                created_at=datetime.fromisoformat(task_data["created_at"]),
                completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data["completed_at"] else None
            )
            plan.tasks[task_id] = task
        
        # Reconstruct context summaries
        plan.context_summaries = [
            ContextSummary.from_dict(ctx_data) 
            for ctx_data in data["context_summaries"]
        ]
        
        return plan
