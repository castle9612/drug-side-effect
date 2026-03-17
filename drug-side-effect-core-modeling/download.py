import requests
from bs4 import BeautifulSoup

def get_target_names(drug_id):
    url = f"https://go.drugbank.com/targets?approved=0&nutraceutical=0&illicit=0&investigational=0&withdrawn=0&experimental=0&us=0&ca=0&eu=0&q%5Bdrug%5D={drug_id}&q%5Bassociation_type%5D=target&q%5Bpolypeptides.name%5D=&button="
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract target names from the table
    target_names = []
    table = soup.find('table', {'id': 'targets-table'})
    if table:
        for row in table.tbody.find_all('tr'):
            target_name = row.find_all('td')[2].a.text.strip()
            target_names.append(target_name)

    return target_names

# Example usage
drug_id = 'DB06736'  # Replace with the desired drug ID
names = get_target_names(drug_id)
for name in names:
    print(name)
