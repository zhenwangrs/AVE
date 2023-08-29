import pandas as pd

if __name__ == '__main__':
    csv = pd.read_csv('C:/Users/NINGMEI/Documents/Research/AVE/data/AVE_Dataset/trainSet.txt', sep='&')
    # 取第二列
    # video_ids = csv['VideoID'].tolist()
    video_ids = csv.iloc[:, 1]
    id_dict = {}
    for video_id in video_ids:
        if video_id not in id_dict:
            id_dict[video_id] = 1
        else:
            id_dict[video_id] += 1
    # 输出重复的视频id
    for video_id in id_dict:
        if id_dict[video_id] > 1:
            print(video_id)
    pass
