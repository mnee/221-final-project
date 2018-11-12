import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id="c155fe505c654167966a78cf97bd0423",
        client_secret="7b2c4ca8d4744e94ad276639ed71cac9")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

song_uris = ["spotify:track:7qiZfU4dY1lWllzX7mPBI","spotify:track:5CtI0qwDJkDQGwXD1H1cL","spotify:track:4aWmUDTfIPGksMNLV2rQP","spotify:track:6RUKPb4LETWmmr3iAEQkt","spotify:track:3DXncPQOG4VBw3QHh3S81","spotify:track:7KXjTSCq5nL1LoYtL7XAw","spotify:track:3eR23VReFzcdmS7TYCrhC","spotify:track:3B54sVLJ402zGa6Xm4YGN","spotify:track:0KKkJNfGyhkQ5aFogxQAP","spotify:track:3NdDpSvN911VPGivFlV5d","spotify:track:7GX5flRQZVHRAGd6B4TmD","spotify:track:72jbDTw1piOOj770jWNea","spotify:track:0dA2Mk56wEzDgegdC6R17","spotify:track:4iLqG9SeJSnt0cSPICSjx","spotify:track:0VgkVdmE4gld66l8iyGjg","spotify:track:3a1lNhkSLSkpJE4MSHpDu","spotify:track:6kex4EBAj0WHXDKZMEJaa","spotify:track:6PCUP3dWmTjcTtXY02oFd","spotify:track:5knuzwU65gJK7IF5yJsua","spotify:track:0CcQNd8CINkwQfe1RDtGV","spotify:track:2rb5MvYT7ZIxbKW5hfcHx","spotify:track:0tKcYR2II1VCQWT79i5Nr","spotify:track:5uCax9HTNlzGybIStD3vD","spotify:track:79cuOz3SPQTuFrp8WgftA","spotify:track:6De0lHrwBfPfrhorm9q1X","spotify:track:6D0b04NJIKfEMg040WioJ","spotify:track:0afhq8XCExXpqazXczTSv","spotify:track:3ebXMykcMXOcLeJ9xZ17X","spotify:track:7BKLCZ1jbUBVqRi2FVlTV","spotify:track:1x5sYLZiu9r5E43kMlt9f"]
features =sp.audio_features(["spotify:track:5N5k9nd479b1xpDZ4usjrg","spotify:track:47zREtxQZ3cHHIZwUQnuuN"])
print(features)
