def local_vid(name):
    return "data_raw/video/" + name + ".mp4";
def local_pic(name,folder=""):
    if folder == "":
        return "data_raw/picture/" + name +".jpg";
    else:
        return "data_raw/picture/" + folder +"/"+ name +".jpg";

print(local_pic("name","555"))