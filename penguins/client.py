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
    print("\nEjecutando solicitud POST...")
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
curl --request POST "http://127.0.0.1:8000/predict" \
--header "Content-Type: application/json" \
--data-raw "{\
    \"island\": \"Torgersen\",\
    \"sex\": \"FEMALE\",\
    \"bill_length_mm\": 39.1,\
    \"bill_depth_mm\": 18.7,\
    \"flipper_length_mm\": 181.0,\
    \"body_mass_g\": 3800.0\
}"


{
  "penguin": false,
  "penguin_probability": 0.0025385558731211982
}


"""