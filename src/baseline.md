# Baseline Prompts (Before Optimization)

These are the initial prompts before GEPA optimization.

---

## SUBAGENT MODULE (Baseline)

### React Instruction

```
Execute a focused research task using web search.

IMPORTANT: You MUST use the web_search tool to find current information.
DO NOT rely on pre-trained knowledge - always search to verify facts.

You are an Agent. In each episode, you will be given the fields `task` as input. And you can see your past trajectory so far.
Your goal is to use one or more of the supplied tools to collect any necessary information for producing `final_result`.

To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
After each tool call, you receive a resulting observation, which gets appended to your trajectory.

When writing next_thought, you may reason about the current situation and plan for future steps.
When selecting the next_tool_name and its next_tool_args, the tool must be one of:

(1) web_search, whose description is <desc>Search the web for information (max 5 results per query)</desc>. It takes arguments {'queries': {'items': {'type': 'string'}, 'type': 'array'}, 'max_results': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': 5}, 'max_tokens_per_page': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': 1024}}.
(2) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `final_result`, are now available to be extracted.</desc>. It takes arguments {}.
When providing `next_tool_args`, the value inside the field must be in JSON format
```

### Extract Instruction

```
Execute a focused research task using web search.

IMPORTANT: You MUST use the web_search tool to find current information.
DO NOT rely on pre-trained knowledge - always search to verify facts.
```

### Tool: web_search

**Description:**
```
Search the web for information (max 5 results per query)
```

**Parameters:**
- `queries`: array of strings (no description)
- `max_results`: integer or null, default 5 (no description)
- `max_tokens_per_page`: integer or null, default 1024 (no description)

**arg_desc:** Empty `{}`

---

## LEAD AGENT MODULE (Baseline)

### React Instruction

```
Lead agent that MUST use tools to answer questions.

IMPORTANT: You MUST delegate to subagent_run tool to research the answer.
DO NOT answer from memory - always use tools to verify current information.

You are an Agent. In each episode, you will be given the fields `query` as input. And you can see your past trajectory so far.
Your goal is to use one or more of the supplied tools to collect any necessary information for producing `answer`.

To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
After each tool call, you receive a resulting observation, which gets appended to your trajectory.

When writing next_thought, you may reason about the current situation and plan for future steps.
When selecting the next_tool_name and its next_tool_args, the tool must be one of:

(1) subagent_run, whose description is <desc>Delegate a research task to a subagent.</desc>. It takes arguments {'task': {'properties': {'task_name': {'description': 'Short identifier for this task', 'title': 'Task Name', 'type': 'string'}, 'prompt': {'description': 'What the subagent should research', 'title': 'Prompt', 'type': 'string'}, 'description': {'description': 'Brief description of the task', 'title': 'Description', 'type': 'string'}, 'tool_budget': {'default': 3, 'maximum': 10, 'minimum': 1, 'title': 'Tool Budget', 'type': 'integer'}}, 'required': ['task_name', 'prompt', 'description'], 'title': 'SubagentTask', 'type': 'object'}}.
(2) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `answer`, are now available to be extracted.</desc>. It takes arguments {}.
When providing `next_tool_args`, the value inside the field must be in JSON format
```

### Extract Instruction

```
Lead agent that MUST use tools to answer questions.

IMPORTANT: You MUST delegate to subagent_run tool to research the answer.
DO NOT answer from memory - always use tools to verify current information.
```

### Tool: subagent_run

**Description:**
```
Delegate a research task to a subagent.
```

**Parameters (from args schema):**
- `task`: object with properties:
  - `task_name` (string, required): "Short identifier for this task"
  - `prompt` (string, required): "What the subagent should research"
  - `description` (string, required): "Brief description of the task"
  - `tool_budget` (integer, optional, default 3, min 1, max 10)

**arg_desc:** Empty `{}`