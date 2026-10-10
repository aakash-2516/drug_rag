import xml.etree.ElementTree as ET
import json

tree = ET.parse("metformin.xml")
root = tree.getroot()

namespace = {
    "hl7": "urn:hl7-org:v3"
}

sections = root.findall(".//hl7:section", namespace)

documents = []

for section in sections:

    title = section.find("hl7:title", namespace)
    text = section.find("hl7:text", namespace)

    if title is not None:
        section_title = " ".join(title.itertext()).strip()
    else:
        section_title = "Unknown"

    if text is not None:
        section_text = " ".join(text.itertext()).strip()
    else:
        section_text = ""

    if section_text:
        documents.append({
            "drug": "Metformin hydrochloride",
            "section": section_title,
            "text": section_text
        })

with open("metformin.json", "w", encoding="utf-8") as file:
    json.dump(documents, file, indent=4, ensure_ascii=False)

print("Saved", len(documents), "sections")