import requests
import time

# URL of the ESP32 in AP mode
#sp32_ip = "http://192.168.4.1"  # Update this IP if needed

# Function to send command to ESP32
def Move_Bot(url):
    if url == "1":
        link = "http://192.168.4.1/FORWARD"
    elif url == "2":
        link = "http://192.168.4.1/RIGHT"
    elif url == "3":
        link = "http://192.168.4.1/LEFT"
    response = requests.get(link)
    time.sleep(1)
    return response.text


# Example usage
if __name__ == "__main__":
    print("Robot moving forward")
