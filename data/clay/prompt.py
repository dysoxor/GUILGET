import json


def main():
    train_path = "train.json"
    test_path = "test.json"
    f = open(train_path)
    data = json.load(f)
    f.close()
    for screen in data:
        text = "A high quality app interface with"
        is_start = True
        for rel in screen["relationships"]:
            if not is_start:
                text += ","
            is_start = False
            text += " "
            text += screen["objects"][rel["sub_id"]]["class"]
            text += " "
            text += str(rel["sub_id"])
            text += " "
            text += rel["predicate"]
            text += " "
            text += screen["objects"][rel["obj_id"]]["class"]
            text += " "
            text += str(rel["obj_id"])
        dict_prompt = '{"source": "source/'+str(screen['id'])+'.png", "target": "target/'+str(screen['id'])+'.png", "prompt": "'+text+'"}\n'
        with open("prompt_train.json", "a") as myfile:
            myfile.write(dict_prompt)
        #with open("prompt_train.json", 'a') as file:
        #    json.dump(dict_prompt, file, indent=4)

if __name__ == "__main__":
	main()