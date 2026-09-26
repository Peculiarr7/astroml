"""Smoke tests for the quick_start entry point (issue #1000).

``astroml/quick_start.py`` wires sample data through the ingestion -> graph ->
train pipeline (see ``QUICKSTART_GUIDE.md`` and
``docs/api/usage-examples.md``), but had no automated test verifying it
actually runs end-to-end without crashing. The pipeline's first two steps
(``generate_sample_ledgers`` and ``build_sample_graph``) depend directly on
``astroml/db/session.py`` for the database session used to write and read
back sample data, which is this issue's stated area.

This module covers two things:

1. ``TestGetSessionSmoke`` -- a real, fully-passing smoke test of the
   ``astroml.db.session`` layer that ``quick_start`` relies on: that
   ``get_session()``/``get_engine()`` produce a working SQLAlchemy session
   against a database resolved from ``ASTROML_DATABASE_URL``, that the
   session can create the ORM schema and round-trip the exact models
   ``quick_start.py`` uses (``Account``, ``Asset``, ``Ledger``, ``Operation``,
   ``Transaction``), and that ``get_session()`` returns an independently
   usable session on each call.

2. ``TestQuickStartPipelineSmoke`` -- an end-to-end smoke test that actually
   calls ``generate_sample_ledgers`` and ``build_sample_graph`` from
   ``astroml.quick_start`` against a sqlite database and asserts on their
   output (ledger/account counts, edge count, graph shape), matching what
   ``run_quickstart()`` itself does for its first two steps.

Known pre-existing blockers for TestQuickStartPipelineSmoke
-------------------------------------------------------------
Two independent, pre-existing problems currently prevent
``TestQuickStartPipelineSmoke`` from passing. Both are out of scope for this
smoke-test issue (fixing them touches ``astroml/cache/graph_cache.py`` and
``astroml/quick_start.py``'s data-generation logic respectively, neither of
which this issue's acceptance criteria call for), but a smoke test's whole
purpose is to surface exactly this kind of drift, so both are disclosed here
rather than worked around:

1. Import-time: ``astroml/quick_start.py`` imports ``from .benchmarking.config
   import BenchmarkConfig`` and ``from .benchmarking.core import
   BenchmarkResult`` at module level (lines 25-26). ``astroml.benchmarking``
   eagerly imports ``astroml.benchmarking.core``, which imports
   ``astroml.models``, which imports ``astroml.models.link_prediction``,
   which imports ``astroml.cache``. ``astroml/cache/graph_cache.py`` has a
   genuine, pre-existing syntax error (a stray em-dash character breaking a
   docstring, unrelated to this change) that makes ``astroml.cache`` --
   and therefore ``astroml.quick_start`` itself -- fail to import on a clean
   checkout of ``main``. This is the same pre-existing, already-documented
   blocker described in ``tests/test_optimization_issues_765_766_767_768.py``
   (already merged, on ``main``): tests needing an import path through
   ``astroml.cache`` fail with the identical
   ``SyntaxError: invalid character '—' (U+2014)``.

2. Runtime, independent of (1): even once ``astroml.cache`` imports cleanly,
   ``generate_sample_ledgers()`` in ``astroml/quick_start.py`` constructs ORM
   rows using field names that no longer match the current schema in
   ``astroml/db/models/__init__.py`` -- e.g. ``Asset(code=..., issuer=...)``
   where the model's columns are ``asset_code``/``asset_issuer``, and
   ``Account(id=...)`` where the model's primary key column is
   ``account_id``. ``astroml/db/models/__init__.py`` has been revised several
   times more recently than ``astroml/quick_start.py`` (last touched in
   ``25e422e``), so ``quick_start.py`` appears to have drifted out of sync
   with schema changes rather than the schema being wrong. This is a second,
   separate, pre-existing bug this smoke test surfaces; fixing the
   schema/quick_start mismatch is a product judgment call (which side should
   change) that belongs to a follow-up issue, not this one.

``TestQuickStartPipelineSmoke`` below is written to be correct against
``generate_sample_ledgers``'s and ``build_sample_graph``'s documented
contracts, and will pass once both (1) and (2) are fixed upstream; until then
it fails for these two pre-existing, disclosed reasons.
``TestGetSessionSmoke`` does not import ``astroml.quick_start`` (only
``astroml.db.session`` and ``astroml.db.schema``, neither of which touches
``astroml.cache``, and it uses the current, correct model field names), so it
is unaffected by either blocker and passes cleanly today.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from astroml.db.schema import Account, Asset, Base, Ledger, Operation, Transaction


@pytest.fixture
def sqlite_database_url(tmp_path, monkeypatch):
    """Point ``ASTROML_DATABASE_URL`` at a fresh on-disk sqlite file.

    Clears ``get_engine``'s ``lru_cache`` before and after so this test's
    engine can't leak into (or be shadowed by) another test's cached engine.
    """
    from astroml.db.session import get_engine

    db_file = tmp_path / "quick_start_smoke.db"
    url = f"sqlite:///{db_file}"
    monkeypatch.setenv("ASTROML_DATABASE_URL", url)
    get_engine.cache_clear()
    yield url
    get_engine.cache_clear()


class TestGetSessionSmoke:
    """Smoke coverage for astroml.db.session, the layer quick_start needs."""

    def test_get_session_returns_working_session(self, sqlite_database_url):
        """get_session() resolves ASTROML_DATABASE_URL and yields a usable session."""
        from astroml.db.session import get_session

        session = get_session()
        try:
            Base.metadata.create_all(session.get_bind())
            asset = Asset(
                asset_type="credit_alphanum4",
                asset_code="XLM",
                asset_issuer="GNATIVE00000000000000000000000000000000000000000000",
            )
            session.add(asset)
            session.commit()

            fetched = session.execute(select(Asset).where(Asset.asset_code == "XLM")).scalar_one()
            assert fetched.asset_code == "XLM"
        finally:
            session.close()

    def test_get_session_returns_independent_sessions(self, sqlite_database_url):
        """Each get_session() call returns its own Session bound to the shared engine."""
        from astroml.db.session import get_session

        session_a = get_session()
        session_b = get_session()
        try:
            assert session_a is not session_b
            assert session_a.get_bind() is session_b.get_bind()
        finally:
            session_a.close()
            session_b.close()

    def test_get_session_round_trips_quick_start_models(self, sqlite_database_url):
        """The exact models quick_start.py writes/reads can round-trip via get_session().

        Mirrors the shape of astroml.quick_start.generate_sample_ledgers /
        build_sample_graph without requiring the full quick_start module
        (which is blocked by the pre-existing astroml.cache import chain,
        see TestQuickStartPipelineSmoke below).
        """
        from datetime import datetime, timezone

        from astroml.db.session import get_session

        session = get_session()
        try:
            Base.metadata.create_all(session.get_bind())

            asset = Asset(
                asset_type="credit_alphanum4",
                asset_code="ASSET0",
                asset_issuer="GISSUER0000000000000000000000000000000000000000000",
            )
            session.add(asset)
            session.commit()

            account = Account(
                account_id="GACCOUNT000000",
                balance=1000.0,
                sequence=0,
                flags=0,
                last_modified_ledger=1,
            )
            session.add(account)
            session.commit()

            ledger = Ledger(
                sequence=1,
                hash="hash_00000001",
                prev_hash=None,
                closed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                successful_transaction_count=1,
                failed_transaction_count=0,
                operation_count=1,
            )
            session.add(ledger)
            session.flush()

            txn = Transaction(
                hash="txn_1_0",
                ledger_sequence=1,
                source_account=account.account_id,
                created_at=ledger.closed_at,
                fee=100,
                operation_count=1,
                successful=True,
                memo="sample_txn_0",
            )
            session.add(txn)
            session.flush()

            # Operation.id is a plain BigInteger primary key (unlike, e.g.,
            # GraphAccount/GraphEdge, which use a sqlite-variant Integer for
            # this exact reason), so sqlite's autoincrement rowid handling
            # does not kick in for it through the ORM. Set it explicitly
            # here; this test is about get_session() round-tripping, not
            # about exercising autoincrement semantics.
            operation = Operation(
                id=1,
                transaction_hash=txn.hash,
                application_order=1,
                type="payment",
                source_account=account.account_id,
                destination_account=account.account_id,
                amount=42.0,
                asset_code=asset.asset_code,
                asset_issuer=asset.asset_issuer,
                created_at=ledger.closed_at,
            )
            session.add(operation)
            session.commit()

            operations = session.execute(select(Operation)).scalars().all()
            assert len(operations) == 1
            assert operations[0].amount == pytest.approx(42.0)
        finally:
            session.close()


class TestQuickStartPipelineSmoke:
    """End-to-end smoke test for quick_start's data-generation pipeline.

    See module docstring: blocked by the pre-existing astroml.cache
    SyntaxError, same precedent as tests/test_optimization_issues_765_766_767_768.py.
    """

    @pytest.fixture
    def db_session(self, tmp_path):
        db_file = tmp_path / "quick_start_pipeline.db"
        engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        session = session_local()
        yield session
        session.close()
        engine.dispose()

    def test_generate_sample_ledgers_and_build_graph_smoke(self, db_session):
        """generate_sample_ledgers + build_sample_graph run without crashing.

        Uses a small sample size (5 ledgers / 4 accounts / 2 assets) since
        this is a smoke test, not a benchmark; tests/performance covers
        performance separately.
        """
        from astroml.quick_start import build_sample_graph, generate_sample_ledgers

        ledger_sequences, account_ids = generate_sample_ledgers(
            db_session,
            num_ledgers=5,
            num_accounts=4,
            num_assets=2,
            txns_per_ledger=3,
        )

        assert len(ledger_sequences) == 5
        assert len(account_ids) == 4

        edges, node_index = build_sample_graph(db_session, ledger_sequences, account_ids)

        assert len(edges) == 15  # 5 ledgers * 3 txns/ledger
        assert len(node_index) == 4
        assert set(node_index.keys()) == set(account_ids)
        for edge in edges:
            assert edge.src in account_ids
            assert edge.dst in account_ids
            assert edge.src != edge.dst  # generate_sample_ledgers avoids self-loops
