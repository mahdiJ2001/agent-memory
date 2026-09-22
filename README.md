# Agent Memory

![Architecture diagram](images/architecture-diagram.png)

A small shopping assistant built to demonstrate, side by side, what memory actually adds to an LLM agent. The same agent, the same tools, the same model - run once with memory disabled and once with memory enabled - to make the difference observable rather than theoretical.

This is a learning project. The product (a tiny shoe store with products, customers, and orders) is deliberately minimal; the point is the agent harness and its memory system, not the e-commerce logic.

## The core question

A plain LLM call answers from the current conversation only. Close the session and everything is gone - preferences you stated, things that happened, lessons learned. This project builds an agent harness around a base LLM and adds three kinds of memory on top of it, then demonstrates the value of each one with a controlled before/after comparison: identical customer, identical question, only the memory system toggled.

```
                SIMPLE E-COMMERCE APP
                         |
                         v
                   AGENT HARNESS
                         |
              +----------+----------+
              v          v          v
           LLM API     Tools      Memory
                                     |
                        +------------+------------+
                        v            v             v
                    Semantic     Episodic      Procedural
```

## The three types of memory

- **Semantic memory** - general, durable facts/preferences about a customer, detached from any specific occasion. E.g. "Prefers lightweight running shoes."
- **Episodic memory** - a specific past event with a concrete outcome, tied to real data (an order). E.g. "Bought the Nike Pegasus, returned it, felt too narrow."
- **Procedural memory** - a global, step-by-step process for how the agent should carry out a task, not tied to any one customer, and revisable from feedback.

Semantic and episodic memory can look similar, but semantic is a **distilled generalization** ("prefers wider shoes"), while episodic is a **specific, checkable instance** ("returned the Pegasus, too narrow") - the latter is more concrete and persuasive, and the two can even point in different directions at once (generally likes Nike, but this one Nike model didn't work out).

## Architecture

- **LLM**: Claude (`claude-haiku-4-5`) via the Anthropic Messages API
- **Tools**: `search_products`, `get_product`, `create_order`, `get_order`, plus memory-specific tools described below
- **Database**: PostgreSQL with the `pgvector` extension, run via Docker Compose
- **Embeddings**: local, via `sentence-transformers` (`all-MiniLM-L6-v2`) - no external API, no cost, works offline
- **Interface**: a CLI (`uv run agent-memory`)

## Semantic memory

Semantic memory holds durable facts and preferences about a customer, independent of any specific event - the kind of thing that stays true across many conversations.

- **Writing**: model-driven. The agent has a `remember_user_fact` tool and decides on its own, the same way it decides to call any tool, when something the customer says is worth keeping.
- **Reading**: harness-driven. Before every message reaches the model, the harness embeds it, searches `semantic_memories` in pgvector for that customer's closest facts, and prepends them to the message as a `[Known facts about this customer]` block. The model never asks for this - it just appears in context.

**Without memory** - the agent is told the customer's running distance and favorite shoe, but the moment the session ends, that information is gone:

![No memory: preferences stated](images/Nomemory.png)

![No memory: asked again in a new session, agent has no idea](images/nomemory2.png)

**With memory** - the same preferences are stated once:

![With memory: preferences stated](images/semanticmemory1.png)

...and correctly recalled in a brand new session, without the customer repeating anything:

![With memory: recalled correctly in a new session](images/semanticmemory2.png)

The underlying `semantic_memories` table in Postgres, holding the embedded rows that made this possible:

![semantic_memories table](images/semanticmemorypg.png)

## Episodic memory

Episodic memory holds specific past events and their outcomes, tied to real data (an order), rather than a generalized preference. "Prefers wider shoes" is a fact; "bought the Pegasus, returned it, it felt too narrow" is an episode - concrete, falsifiable, and far more persuasive when the agent brings it back up later.

- **Writing**: event-derived. There is no "remember this episode" tool. Instead, the agent has a `return_order` tool for processing an actual return. As a side effect of that real action, the harness automatically writes an episodic memory - the model only decided to process a return, not to save a memory.
- **Reading**: harness-driven, identical mechanism to semantic memory - embed the current message, search `episodic_memories` for that customer, inject the closest matches as a `[Past experiences with this customer]` block.

The customer buys a Nike Pegasus, then returns it because it felt too narrow:

![Ordering then returning the Nike Pegasus](images/episodic1.png)

**Without memory**, asked for running shoe recommendations afterward, the agent still suggests the Nike Pegasus again - it has no record the return ever happened:

![No memory: Pegasus recommended again despite the return](images/episodic2.png)

**With memory**, the same request retrieves the episode and the agent avoids repeating the mistake, steering toward a wider-fitting alternative instead:

![With memory: Pegasus is not recommended again](images/episodic3.png)

The underlying `episodic_memories` table, holding the order return that was recorded automatically:

![episodic_memories table](images/pgepisodic.png)

## Procedural memory

Procedural memory holds a stored, step-by-step process for how the agent should carry out a task - not a fact about a customer, but an instruction about behavior. It is global rather than per-customer: one person's feedback about *how the agent should operate* changes it for everyone.

- **Writing**: model-driven, via an `update_procedure` tool, called when a customer's feedback implies the process itself should permanently change - not just this one answer.
- **Reading**: harness-driven. The current procedure is fetched fresh on every turn and injected as a `[Procedure to follow when recommending running shoes]` block, so a mid-conversation revision takes effect immediately.

The seeded procedure for recommending running shoes:

1. Ask about the customer's running distance, if not already known.
2. Ask whether they prefer lightweight or cushioned shoes, if not already known.
3. Check the customer's previous purchases.
4. Check the customer's previous returns.
5. Recommend 2-3 products.

The agent following those steps - asking before recommending, rather than guessing:

![Agent following the stored procedure](images/proceduralmemory.png)

The `procedures` table in Postgres, holding the steps as they currently stand:

![procedures table](images/preceduralPg.png)

Note step 1 and step 2 are conditioned on "if not already known" - procedural memory is written to defer to semantic memory rather than operate in isolation. If a customer already has their distance and preference stored as facts, the procedure correctly skips asking again. The memory types compose; they are not siloed.

## Project structure

```
src/agent_memory/
  cli.py              interactive loop, memory injection, tool-call loop
  llm.py              Claude client, thin Chat/Response wrapper over the Messages API
  tools.py            tool declarations and dispatch (catalog, orders, memory-writing)
  memory.py           embedding, semantic + episodic read/write helpers
  db.py               schema, seed data, all SQL
docker-compose.yml     Postgres + pgvector
```
