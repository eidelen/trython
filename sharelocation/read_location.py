from locationsharinglib import Service
import time

cookies_file = 'google.com_cookies.txt'
google_email = 'eidelen7@gmail.com'
path_to_pos_file = "d:\\MAK\\vrforces5.0.3\\bin64\\poses.txt"

service = Service(cookies_file=cookies_file, authenticating_account=google_email)


while True:
    pos_file = open(path_to_pos_file, "w")
    pos_text = ""

    for person in service.get_all_people():
        pos_text = pos_text + str(person.latitude) + "    " + str(person.longitude) + "\n"

    pos_file.write(pos_text)

    print(pos_text)

    pos_file.close()
    time.sleep(5)

