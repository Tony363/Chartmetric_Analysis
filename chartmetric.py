import requests,json
import pandas as pd
import numpy as np

"""
Refresh Token for this account is mx8CBeBqlw0dzPKo7K9RNglLw1xKEqQdRagXFmg8DYjtoPocX25iwSDFXGA3c2Jn.
Retrieve your access token with this command:
[curl -d '{"refreshtoken":"mx8CBeBqlw0dzPKo7K9RNglLw1xKEqQdRagXFmg8DYjtoPocX25iwSDFXGA3c2Jn"}' -H "Content-Type: application/json" -X POST https://api.chartmetric.com/api/token]
You can find our official API docs here.

curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NjkzNTUsInRpbWVzdGFtcCI6MTYwNDY2NDg5NDQyNCwiaWF0IjoxNjA0NjY0ODk0LCJleHAiOjE2MDQ2Njg0OTR9.ECXfmUpCJfGoTWDEYMca3JRHy99pmhHzNYOw93h1Ky4" https://api.chartmetric.com/api/album/5449730/spotify/current/playlists
"""

refresh_token_url = 'https://api.chartmetric.com/api/token'
headers = {"refreshtoken":"mx8CBeBqlw0dzPKo7K9RNglLw1xKEqQdRagXFmg8DYjtoPocX25iwSDFXGA3c2Jn"}
post= requests.post(refresh_token_url,data=headers)
res = json.loads((post.content).decode("UTF-8"))
access_token = res['token']
auth = {'Authorization':"Bearer {access_token}".format(access_token=access_token)}
"""
constant query of data saved 
getting a snap shot of all artists montly listeners over a 6 month period
or get time series of month for 200 or so artists and run 200 for 200 diff IDs(provided)

"""
getartist = "https://api.chartmetric.com/api/artist/sp_followers/list?min=5000&max=10000&offset=100000"
# artist = json.loads(requests.get(getartist, headers=auth).content)
# print(artist[0])

"""
time series by artist id
add artist name using get artist id
"""
massive = []
for idx,id in enumerate(np.random.randint(low=100, high=999, size=200)):
    print("Artist ID:",id,'\n')
    try:
        gettime = f"https://api.chartmetric.com/api/artist/{id}/stat/spotify?since=2020-01-01&until=2020-06-01"
        artist_name = f"https://api.chartmetric.com/api/artist/chartmetric/{id}/get-ids"
        name = json.loads(requests.get(artist_name, headers=auth).content)
        data = json.loads(requests.get(gettime, headers=auth).content)
        print(data['obj'].keys())
        print(name['obj'][0].keys())
        lst = []
      
        for subjects in data['obj'].keys():
            topic = data['obj'][subjects]
            if type(topic) == str:
                print(topic)
            else:
                dnew = {}
                for idx,dic in enumerate(topic):
                    for key in dic.keys():
                        if key == "diff":
                            continue
                        if idx == 0:
                            dnew[key] = [dic[key]]
                        else:
                            dnew[key].append(dic[key])
                df = pd.DataFrame(dnew)
                df['Subject'] = pd.Series(np.asarray([subjects for i in range(df.shape[0])]),index=df.index)
                df = df.set_index("Subject")
                df.reset_index(inplace=True)
                lst.append(df)
       
        huge = pd.concat(lst,axis=1)
        huge["Chartmetric_ID"] = pd.Series(np.asarray([name['obj'][0]['artist_name'] for i in range(huge.shape[0])]), index=huge.index)
        huge = huge.set_index("Chartmetric_ID")
        print(huge)
    #    huge.to_csv(f"~/Desktop/{id}_{name['obj'][0]['artist_name']}.csv")
        massive.append(huge)
    except Exception as e:
        print(e)
        massize = pd.concat(huge,axis=0)
        massive.to_csv(f"{id}_chartmetric.csv")
        continue
    
massize = pd.concat(huge,axis=0)
massive.to_csv(f"chartmetric.csv")
  


