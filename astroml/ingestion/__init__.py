"""Data ingestion module for AstroML.

This module handles ingestion of Stellar network ledger data including:
- Incremental ledger processing with idempotency guarantees
- State management for tracking processed ledgers
- Streaming and batch ingestion modes
- Integration with Stellar Horizon API

Key components:
- IngestionService: Main service for ledger ingestion
- StateStore: Persistent state management
- Enhanced streaming ingestion with backpressure control

Dependencies:
- stellar-sdk: Stellar Horizon API client
- aiohttp: Async HTTP client for streaming
"""

from astroml.ingestion.horizon_stream import HorizonStreamError, HorizonStreamingClient

__all__ = ["HorizonStreamingClient", "HorizonStreamError"]
