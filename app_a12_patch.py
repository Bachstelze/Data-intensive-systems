"""One-time helper to insert the A12 Gradio tab into the existing app.py.

Run from the repository root:
    python3 app_a12_patch.py

The script is intentionally small and conservative: it adds one import and one
new tab, but does not rewrite the rest of app.py.
"""

from pathlib import Path

APP_PATH = Path("app.py")

IMPORT_LINE = "from A12.service.ui import run_a12_tab\n"
AFTER_IMPORT = "from A12.pose_interpolator import smooth_pose_sequence\n"

TAB_CODE = '''

        # A12 Classifier Tab
        with gr.TabItem("🧪 A12 Classifier"):
            gr.Markdown(
                """
                ### A12 Service Endpoint: Pose CSV classifier

                Endpoint alternative chosen: **Gradio tab inside the existing app**.
                This keeps the HuggingFace Space architecture simple and avoids a
                separate REST/FastAPI service. Upload a pose-feature CSV exported
                with the same feature schema used by the A12 classifier.
                """
            )
            with gr.Row():
                with gr.Column():
                    a12_csv_input = gr.File(
                        label="Pose feature CSV",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    a12_problem_input = gr.Radio(
                        choices=["A", "B"],
                        value="B",
                        label="Classifier problem",
                        info="A = Kinect 3D features, B = PoseNet 2D features",
                    )
                    a12_predict_btn = gr.Button("Run A12 classifier", variant="primary")

                with gr.Column():
                    a12_summary_output = gr.Markdown(label="Summary")
                    a12_json_output = gr.JSON(label="Structured JSON output")

            a12_predict_btn.click(
                fn=run_a12_tab,
                inputs=[a12_csv_input, a12_problem_input],
                outputs=[a12_json_output, a12_summary_output],
            )
'''

INSERT_BEFORE = "    # Example section\n"


def main() -> None:
    text = APP_PATH.read_text()

    if IMPORT_LINE not in text:
        if AFTER_IMPORT not in text:
            raise RuntimeError("Could not find A12 pose_interpolator import line.")
        text = text.replace(AFTER_IMPORT, AFTER_IMPORT + IMPORT_LINE, 1)

    if "with gr.TabItem(\"🧪 A12 Classifier\")" not in text:
        if INSERT_BEFORE not in text:
            raise RuntimeError("Could not find location for the A12 tab insertion.")
        text = text.replace(INSERT_BEFORE, TAB_CODE + "\n" + INSERT_BEFORE, 1)

    APP_PATH.write_text(text)
    print("A12 Gradio tab inserted into app.py")


if __name__ == "__main__":
    main()
