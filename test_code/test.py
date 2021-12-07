import tweepy

consumer_key = '3h7gTcpHsJqUVEpDfyiaFtk2c'
consumer_secret = 'uBEcCZbwZecgnTVWEhAMw3EIdRxps8Zx5cHOOM2ldrG8R1txrT'
access_token = '1059852994731106304-nZPxMcG0TbutEhfIFqeWdv4eVTrvLa'
access_token_secret = 'Wd3X87eeF65SsetcWSHIEz22JorutzzWt4SyTzEUfsqaQ'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

