"""
Branch Queue — Banco Meridional Terminal-Based Queue Manager
==============================================================

Design notes
------------

1. Why a separate queue per service type makes call_next more efficient
   than a single shared queue:

   With a single ordered list of ALL waiting clients, an agent calling
   their next client would need to scan from the front until they find
   a ticket matching their service type.  In the worst case (many
   clients of other service types arrived earlier) this scan is O(N)
   in the total queue length.  By maintaining one deque per service
   type, each call_next becomes O(1) — we simply popleft from the
   relevant deque, regardless of how many people are waiting for
   other services.  This matches the real-world behaviour of having
   dedicated agents who don't interfere with each other.

2. What would happen if two agents of the SAME service type called
   call_next at the same moment, and what state mutation must happen
   first to ensure the same client is not called twice?

   Two concurrent call_next calls for the same service type would both
   try to read (and remove) the head of that service deque.  Without
   synchronisation they could both see the same ticket, leading to
   the same client being called twice.  The critical invariant is
   that the dequeue (pop) MUST complete before either agent proceeds
   to interact with the client.  In a single-threaded Python script
   this race does not arise, but in a multi-threaded / multi-process
   deployment the deque pop must be protected by a lock (or the
   equivalent atomic operation) that serialises concurrent readers —
   whoever acquires the lock first pops the ticket; the second caller
   sees the next ticket (or an empty queue).
"""

import datetime
from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_SERVICE_TYPES = ("deposit", "withdrawal", "account_management")


@dataclass
class Ticket:
    """A single service ticket held by a waiting client."""

    number: int
    client_name: str
    service_type: str
    issued_at: datetime.datetime = field(default_factory=datetime.datetime.now)


# ---------------------------------------------------------------------------
# Branch queue logic
# ---------------------------------------------------------------------------

class BranchQueue:
    """Manages one FIFO deque per service type plus a global ticket counter."""

    def __init__(self):
        self._queues: dict[str, deque[Ticket]] = {
            st: deque() for st in VALID_SERVICE_TYPES
        }
        self._counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def issue_ticket(self, client_name: str, service_type: str) -> Ticket:
        """Create and enqueue a ticket; return it."""
        self._validate_service_type(service_type)
        self._counter += 1
        ticket = Ticket(
            number=self._counter,
            client_name=client_name,
            service_type=service_type,
        )
        self._queues[service_type].append(ticket)
        return ticket

    def call_next(self, service_type: str) -> Ticket:
        """Dequeue and return the next waiting ticket for *service_type*.

        Raises ``IndexError`` (with a clear message) when the queue is
        empty.
        """
        self._validate_service_type(service_type)
        q = self._queues[service_type]
        if not q:
            raise IndexError(
                f"No clients waiting for '{service_type}'."
            )
        return q.popleft()

    def peek_next(self, service_type: str) -> Ticket | None:
        """Return the next ticket for *service_type* without removing it.

        Returns ``None`` when the queue is empty.
        """
        self._validate_service_type(service_type)
        q = self._queues[service_type]
        if not q:
            return None
        return q[0]

    def list_waiting(self) -> dict[str, list[Ticket]]:
        """Return a dict mapping each service type to its ordered tickets."""
        return {
            st: list(q) for st, q in self._queues.items()
        }

    def stats(self) -> dict:
        """Return per-service counts plus a ``total`` key."""
        counts = {
            st: len(q) for st, q in self._queues.items()
        }
        counts["total"] = sum(counts.values())
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_service_type(service_type: str) -> None:
        if service_type not in VALID_SERVICE_TYPES:
            raise ValueError(
                f"Invalid service type '{service_type}'. "
                f"Valid options: {', '.join(VALID_SERVICE_TYPES)}."
            )


# ============================================ ============================
# CLI loop
# ============================================ ============================

_SERVICE_PROMPT = (
    "Enter service type "
    f"({'/'.join(VALID_SERVICE_TYPES)}): "
)


def _menu() -> str:
    print("\n--- Branch Queue Manager ---")
    print("1  Issue a new ticket")
    print("2  Call next client")
    print("3  Peek next client")
    print("4  List all waiting (grouped by service)")
    print("5  Queue stats")
    print("6  Exit")
    return input("Your choice (1-6): ").strip()


def _read_service_type() -> str:
    while True:
        st = input(_SERVICE_PROMPT).strip().lower()
        if st in VALID_SERVICE_TYPES:
            return st
        print(
            f"Invalid service type '{st}'. "
            f"Please enter one of: {', '.join(VALID_SERVICE_TYPES)}."
        )


def _format_ticket(t: Ticket) -> str:
    return (
        f"  #{t.number}  {t.client_name}"
        f"  [{t.service_type}]  "
        f"({t.issued_at.strftime('%H:%M:%S')})"
    )


def run_cli() -> None:
    """Main CLI loop."""
    bq = BranchQueue()

    while True:
        choice = _menu()

        # --- 1  Issue ticket ------------------------------------------------
        if choice == "1":
            name = input("Client name: ").strip()
            if not name:
                print("Client name cannot be empty.")
                continue
            st = _read_service_type()
            ticket = bq.issue_ticket(name, st)
            print(f"Ticket issued: #{ticket.number}  {ticket.client_name}")

        # --- 2  Call next ---------------------------------------------------
        elif choice == "2":
            st = _read_service_type()
            try:
                ticket = bq.call_next(st)
            except IndexError as exc:
                print(exc)
            else:
                print(f"Calling: {_format_ticket(ticket)}")

        # --- 3  Peek next ---------------------------------------------------
        elif choice == "3":
            st = _read_service_type()
            ticket = bq.peek_next(st)
            if ticket is None:
                print(f"No clients waiting for '{st}'.")
            else:
                print(f"Next: {_format_ticket(ticket)}")

        # --- 4  List all waiting --------------------------------------------
        elif choice == "4":
            waiting = bq.list_waiting()
            any_waiting = False
            for st, tickets in waiting.items():
                if not tickets:
                    continue
                any_waiting = True
                print(f"\n  --- {st} ({len(tickets)} waiting) ---")
                for t in tickets:
                    print(f"    #{t.number}  {t.client_name}  ({t.issued_at.strftime('%H:%M:%S')})")
            if not any_waiting:
                print("No clients currently waiting.")

        # --- 5  Stats -------------------------------------------------------
        elif choice == "5":
            s = bq.stats()
            print(f"  deposit:              {s['deposit']}")
            print(f"  withdrawal:           {s['withdrawal']}")
            print(f"  account_management:   {s['account_management']}")
            print(f"  ---------------------------")
            print(f"  total:                {s['total']}")

        # --- 6  Exit --------------------------------------------------------
        elif choice == "6":
            print("Goodbye!")
            break

        # --- Invalid choice -------------------------------------------------
        else:
            print("Invalid choice. Please enter 1-6.")


# ============================================ ============================
# Entry point
# ============================================ ============================

if __name__ == "__main__":
    run_cli()