import requests

set_id = "014f2a32-8440-4d0c-8f5d-c4b3a49e0dec"

url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"

response = requests.get(url)

print("Status:", response.status_code)

if response.status_code == 200:

    with open("metformin.xml", "w", encoding="utf-8") as file:
        file.write(response.text)

    print("XML saved successfully!")

else:
    print("Failed to download XML")