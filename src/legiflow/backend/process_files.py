import yaml
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from backend.juri_flow import build_llm



with open("prompts.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

metadata = config["prompts"]["metadata"]

metadata_prompt = (
    f"Description: {metadata['description']}\n"
    f"Types de documents:\n- " + "\n- ".join(metadata["document_types"])
)

llm = build_llm()


class Metadata(BaseModel):
    first_name: str
    last_name: str
    document_type: str
    ocr_data: str

    def to_path(self):
        return f"{self.document_type}/{self.first_name}_{self.last_name}"

def process_user_file(user_input):
    files_in_chat = []
    for f in user_input.files:
        if bytes_data := f.read():
            human_content = [
                {"type": "text", "text": metadata_prompt}, 
                {"type": "media", "data": bytes_data, "mime_type": f.type}
            ]
            structured_output = llm.with_structured_output(Metadata)
            answer: Metadata = structured_output.invoke([HumanMessage(content=human_content)])
            relative_path = answer.to_path()
            print("final_answer : ", answer)
            metadata_json = answer.model_dump_json()
            files_in_chat.append(metadata_json) 
            BASE_DIR = Path("/home/massilia/Documents")
            # get path
            final_dir = BASE_DIR / relative_path
            final_dir.mkdir(parents=True, exist_ok=True) 
            # use the bytes, file name and file type to save the file in the predicted path.
            file_name = f.name
            save_path = final_dir / file_name
            with open(save_path, "wb") as out:
                out.write(bytes_data)

            print("Saved to:", save_path)
    return files_in_chat