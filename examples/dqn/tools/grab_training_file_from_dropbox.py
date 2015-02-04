# Include the Dropbox SDK
import dropbox
import os
import sys

sys.path.insert(0, os.path.pardir)
import secrets
from dateutil.parser import parse as dateparse

# Get your app key and secret from the Dropbox developer website
app_key    = secrets.DROPBOX_APP_KEY
app_secret = secrets.DROPBOX_APP_SECRET


def get_latest():
    if secrets.DROPBOX_TOKEN:
        access_token = secrets.DROPBOX_TOKEN
    else:
        flow = dropbox.client.DropboxOAuth2FlowNoRedirect(app_key, app_secret)

        # Have the user sign in and authorize this token
        authorize_url = flow.start()
        print '1. Go to: ' + authorize_url
        print '2. Click "Allow" (you might have to log in first)'
        print '3. Copy the authorization code.'
        code = raw_input("Enter the authorization code here: ").strip()

        # This will fail if the user enters an invalid authorization code
        access_token, user_id = flow.finish(code)

    client = dropbox.client.DropboxClient(access_token)

    # f = open('working-draft.txt', 'rb')
    # response = client.put_file('/magnum-opus.txt', f)
    # print 'uploaded: ', response

    folder_metadata = client.metadata(
        '/src_ns/src_a/caffe/examples/dqn/data/episodes')

    episode_logs = filter(lambda _file: _file['path'].find('episode_log_') >= 0,
                          folder_metadata['contents'])
    latest_log = sorted(episode_logs,
                        key=lambda _file: dateparse(_file['modified']))[-1]


    f = client.get_file(latest_log['path'])
    out = open('latest_log.csv', 'wb')
    out.write(f.read())
    out.close()