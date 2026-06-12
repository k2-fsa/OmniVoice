from __future__ import annotations

from pathlib import Path
from typing import Optional

from omnivoice.audiobook.costing import estimate_chunk_usage, estimate_cost
from omnivoice.audiobook.storage.repository import SecretMetadataRecord, WorkspaceRepository
from omnivoice.audiobook.storage.schema import initialize_workspace_db
from omnivoice.audiobook.storage.secrets import InMemorySecretStore, SecretStoreError


class ApiWorkspaceController:
    def __init__(self, db_path: Path, secret_store: Optional[InMemorySecretStore] = None):
        self.db_path = Path(db_path)
        self.connection = initialize_workspace_db(self.db_path)
        self.repository = WorkspaceRepository(self.connection)
        self.secret_store = secret_store or InMemorySecretStore()

    def api_status(self, provider: str = "openrouter") -> str:
        metadata = self.repository.get_secret_metadata(provider)
        if not metadata or not metadata.configured:
            return "API key is not configured."
        return (
            f"Configured provider: {metadata.provider}\n"
            f"Fingerprint: {metadata.fingerprint}\n"
            f"Storage: {metadata.storage_mode}\n"
            f"Last test: {metadata.last_test_status or 'not tested'}"
        )

    def save_api_key(self, provider: str, api_key: str) -> tuple[str, str]:
        try:
            result = self.secret_store.save(provider, api_key)
        except SecretStoreError as exc:
            return f"Error: {exc}", ""
        self.repository.upsert_secret_metadata(
            SecretMetadataRecord(
                provider=provider,
                fingerprint=result.fingerprint,
                configured=True,
                storage_mode=result.storage_mode,
            )
        )
        return self.api_status(provider), ""

    def remove_api_key(self, provider: str) -> str:
        self.secret_store.remove(provider)
        self.repository.remove_secret_metadata(provider)
        return "API key removed from this local session."

    def estimate(self, text: str, output_tokens: int, input_price: float, output_price: float) -> str:
        usage = estimate_chunk_usage(text or "", expected_output_tokens=int(output_tokens or 0))
        cost = estimate_cost(
            usage,
            input_per_million=float(input_price or 0),
            output_per_million=float(output_price or 0),
        )
        return (
            f"Estimated input tokens: {usage.input_tokens}\n"
            f"Estimated output tokens: {usage.output_tokens}\n"
            f"Estimated total tokens: {usage.total_tokens}\n"
            f"Estimated cost ({cost.currency}): {cost.total_cost:.6f}\n"
            "This is an estimate, not a billing record."
        )

    def close(self) -> None:
        self.connection.close()


def build_api_workspace_page(db_path: Path | str):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gradio is required to launch the API & Costs workspace page") from exc

    controller = ApiWorkspaceController(Path(db_path))
    with gr.Blocks(title="OmniVoice API & Costs") as page:
        gr.Markdown("# API & Costs")
        gr.Markdown("Configure provider access locally. Saved keys are not displayed after save.")
        with gr.Tab("API Key"):
            provider = gr.Textbox(label="Provider", value="openrouter")
            key = gr.Textbox(label="API key", type="password")
            status = gr.Textbox(label="Status", value=controller.api_status(), lines=5)
            with gr.Row():
                save = gr.Button("Save key", variant="primary")
                remove = gr.Button("Remove key")
            save.click(controller.save_api_key, inputs=[provider, key], outputs=[status, key])
            remove.click(controller.remove_api_key, inputs=[provider], outputs=[status])
        with gr.Tab("Token & Cost Simulator"):
            text = gr.Textbox(label="Text or chunk preview", lines=8)
            output_tokens = gr.Number(label="Expected output tokens", value=512)
            input_price = gr.Number(label="Input price per 1M tokens", value=0.0)
            output_price = gr.Number(label="Output price per 1M tokens", value=0.0)
            estimate_button = gr.Button("Estimate")
            estimate_output = gr.Textbox(label="Estimate", lines=7)
            estimate_button.click(
                controller.estimate,
                inputs=[text, output_tokens, input_price, output_price],
                outputs=[estimate_output],
            )
    return page


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the local API & Costs workspace page.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    build_api_workspace_page(args.db).launch(server_name=args.ip, server_port=args.port)


if __name__ == "__main__":
    main()
