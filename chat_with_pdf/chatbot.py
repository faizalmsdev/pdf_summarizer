import os
import tempfile
from pathlib import Path
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import reflex as rx
from embedchain import App

# Set the path to Tesseract if it's not in your system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Styling for messages
message_style = dict(display="inline-block", padding="2em", border_radius="8px",
                     max_width=["120em", "120em", "80em", "80em", "80em", "80em"])

class State(rx.State):
    """The app state."""
    messages: list[dict] = []
    db_path: str = tempfile.mkdtemp()  # Temporary directory for the knowledge base
    pdf_filename: str = ""
    knowledge_base_files: list[str] = []
    user_question: str = ""
    upload_status: str = ""

    def get_app(self):
        """Initialize and return the EmbedChain app."""
        return App.from_config(
            config={
                "llm": {"provider": "ollama",
                        "config": {"model": "llama3.2:latest", "max_tokens": 250, "temperature": 0.5, "stream": True,
                                   "base_url": 'http://localhost:11434'}},
                "vectordb": {"provider": "chroma", "config": {"dir": self.db_path}},
                "embedder": {"provider": "ollama",
                             "config": {"model": "llama3.2:latest", "base_url": 'http://localhost:11434'}},
            }
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess the image for better OCR results."""
        # Convert to grayscale
        gray_image = image.convert("L")
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2.0)  # Increase contrast
        # Optionally apply some filters (e.g., sharpen)
        sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)
        return sharpened_image

    def extract_text_with_ocr(self, pdf_path):
        """Extract text from PDF using OCR with image preprocessing."""
        print(f"Starting OCR extraction for: {pdf_path}")
        extracted_text = ""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                # Preprocess the image
                processed_image = self.preprocess_image(image)
                # Extract text from the processed image using pytesseract
                text = pytesseract.image_to_string(processed_image)
                print(f"Extracted text from page {i + 1}")
                extracted_text += text + "\n\n"
        except Exception as e:
            print(f"Error during OCR: {e}")
        return extracted_text

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the file upload and processing."""
        if not files:
            self.upload_status = "No file uploaded!"
            return

        file = files[0]
        upload_data = await file.read()
        upload_dir = Path(rx.get_upload_dir()) / file.filename  # File path
        self.pdf_filename = file.filename

        # Save the file
        with open(upload_dir, "wb") as file_object:
            file_object.write(upload_data)

        # Check if the file is saved and not empty
        if not os.path.exists(upload_dir) or os.path.getsize(upload_dir) == 0:
            self.upload_status = "File not saved properly or is empty!"
            return

        # Try to extract text normally first
        extracted_text = ""
        try:
            with open(upload_dir, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text
        except Exception as e:
            print(f"Error reading PDF with PyPDF2: {e}")
            self.upload_status = f"Error reading PDF with PyPDF2: {e}"
            return

        # If no text was found, fall back to OCR
        if not extracted_text.strip():
            self.upload_status = f"No selectable text found in {self.pdf_filename}, performing OCR..."
            extracted_text = self.extract_text_with_ocr(upload_dir)
        
        if not extracted_text.strip():
            self.upload_status = f"OCR failed to extract meaningful content from {self.pdf_filename}"
            return

        # Now pass to embedchain for further processing
        app = self.get_app()
        try:
            # Save the extracted text as a temporary text file for embedding
            temp_text_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            with open(temp_text_file.name, "w") as txt_file:
                txt_file.write(extracted_text)
            
            print(f"Uploading file: {file.filename}, Size: {os.path.getsize(upload_dir)} bytes")
            app.add(temp_text_file.name, data_type="text")  # Adding as text
            print("File added successfully")
            self.knowledge_base_files.append(self.pdf_filename)
            self.upload_status = f"Processed and added {self.pdf_filename} to the knowledge base!"
        except ValueError as e:
            print(f"Error adding file: {e}")
            self.upload_status = f"Failed to process {self.pdf_filename}. Error: {e}"

    def chat(self):
        """Handle user questions about the PDF."""
        if not self.user_question:
            return
        app = self.get_app()
        self.messages.append({"role": "user", "content": self.user_question})
        response = app.chat(self.user_question)
        self.messages.append({"role": "assistant", "content": response})
        self.user_question = ""  # Clear the question after sending

    def clear_chat(self):
        """Clear the chat history."""
        self.messages = []


# UI colors
color = "rgb(107,99,246)"


# Define the frontend interface
def index():
    return rx.vstack(
        rx.heading("Chat with PDF using Llama 3.2"),
        rx.text("This app allows you to chat with a PDF using Llama 3.2 running locally with Ollama!"),
        rx.hstack(
            rx.vstack(
                rx.heading("PDF Upload", size="md"),
                rx.upload(
                    rx.vstack(
                        rx.button(
                            "Select PDF File",
                            color=color,
                            bg="white",
                            border=f"1px solid {color}",
                        ),
                        rx.text("Drag and drop PDF file here or click to select"),
                    ),
                    id="pdf_upload",
                    multiple=False,
                    accept={".pdf": "application/pdf"},
                    max_files=1,
                    border=f"1px dotted {color}",
                    padding="2em",
                ),
                rx.hstack(rx.foreach(rx.selected_files("pdf_upload"), rx.text)),
                rx.button(
                    "Upload and Process",
                    on_click=State.handle_upload(rx.upload_files(upload_id="pdf_upload")),
                ),
                rx.button(
                    "Clear",
                    on_click=rx.clear_selected_files("pdf_upload"),
                ),
                rx.text(State.upload_status),  # Display upload status
                width="50%",
            ),
            rx.vstack(
                rx.foreach(
                    State.messages,
                    lambda message, index: rx.cond(
                        message["role"] == "user",
                        rx.box(
                            rx.text(message["content"]),
                            background_color="rgb(0,0,0)",
                            padding="10px",
                            border_radius="10px",
                            margin_y="5px",
                            width="100%",
                        ),
                        rx.box(
                            rx.text(message["content"]),
                            background_color="rgb(0,0,0)",
                            padding="10px",
                            border_radius="10px",
                            margin_y="5px",
                            width="100%",
                        ),
                    )
                ),
                rx.hstack(
                    rx.input(
                        placeholder="Ask a question about the PDF",
                        id="user_question",
                        value=State.user_question,
                        on_change=State.set_user_question,
                        **message_style,
                    ),
                    rx.button("Send", on_click=State.chat),
                ),
                rx.button("Clear Chat History", on_click=State.clear_chat),
                width="50%",
                height="100vh",
                overflow="auto",
            ),
            width="100%",
        ),
        padding="2em",
    )


# Initialize the Reflex app
app = rx.App()
app.add_page(index)
