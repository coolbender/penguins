import requests
import sys

api_url = "http://localhost:8000"

def call_api(island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, endpoint) :

    payload = {
                "island": island,
                "sex": sex,
                "bill_length_mm": bill_length_mm,
                "bill_depth_mm": bill_depth_mm,
                "flipper_length_mm": flipper_length_mm,
                "body_mass_g": body_mass_g
    }
    try:
        response = requests.post(api_url+"/"+endpoint, json=payload)
        print(response.text)

    except requests.RequestException as e:
        print(f"Error de conexión: {e}")

def main(island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, endpoint):
    print(f"\nEjecutando solicitud POST... {endpoint}")
    call_api(island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, endpoint)

if __name__ == "__main__":

    if len(sys.argv) != 8:
        print("Uso: python client.py <island> <sex> <bill_length_mm> <bill_depth_mm> <flipper_length_mm> <body_mass_g> <endpoint>")
        sys.exit(1)

    island = sys.argv[1]
    sex = sys.argv[2]
    bill_length_mm = sys.argv[3]
    bill_depth_mm = sys.argv[4]
    flipper_length_mm = sys.argv[5]
    body_mass_g = sys.argv[6]
    endpoint = sys.argv[7]
  
    main(island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, endpoint)

"""
poetry run python client.py "Torgersen" "male" 39.1 18.7 181.0 3750.0 predict_lr

{
  "penguin (Adelie: 0 | Chinstrap: 1 | Gentoo: 2)": 1,
  "penguin_probability (%)": 50.23

}

"""