import lyric_structure as ls
import pandas as pd
from util import ssm
import os

cnt_list = [-1]

def parse_song_content(song_content):
    # Split song content into lyrics and metadata sections
    lyric_section, metadata_section = song_content.split('__', 1)
    lyrics = lyric_section.strip()
    metadata_section = metadata_section.lstrip("_")

    metadata_lines = metadata_section.strip().split('\n')
    metadata = {}
    for line in metadata_lines:
        key, value = line.split('  ', 1)
        value = value.lstrip()
        metadata[key.lower()] = value

    # Extract desired values from metadata
    name = metadata.get('name', '')
    artist = metadata.get('artist', '')
    album = metadata.get('album', '')
    track_no = metadata.get('track no', '')
    year = metadata.get('year', '')
    cnt_list[0] += 1

    return {
        'id': cnt_list[0],
        'a_lyrics': lyrics,
        'a_name': artist,
        'a_song': name,
        'a_album': album,
        'a_track_no': track_no,
        'a_year': year
    }


# List to store parsed data
songs_data = []

# Step 1: Iterate through each file in the "data" directory
data_folder = 'data'
for file_name in os.listdir(data_folder):
    if file_name.endswith('.txt'):  # Ensure we're reading only text files
        with open(os.path.join(data_folder, file_name), 'r', encoding='utf-8') as file:
            song_content = file.read()
            song_data = parse_song_content(song_content)
            songs_data.append(song_data)

# Step 2: Convert the list to a DataFrame
songs_df = pd.DataFrame(songs_data)
songs_df['borders'] = songs_df['a_lyrics'].apply(ls.segment_borders)

# Create an empty DataFrame for ssms
ssms = pd.DataFrame({'id': songs_df['id'], 'a_lyrics': songs_df['a_lyrics']})
ssms['ssm'] = ssms['a_lyrics'].apply(ls.calculate_ssms)

# Split the ssms dataframe into two
midpoint = len(ssms) // 2
ssms_1 = ssms.iloc[:midpoint]
ssms_2 = ssms.iloc[midpoint:]

# Save the DataFrames
songs_df.to_hdf('resources/mldb_watanabe_5plus_seg_english.hdf', key='mldb_watanabe_5plus_seg_english', mode='w')
ssms_1.to_hdf('resources/ssm_store_pub1.hdf', key='mdb_127_en_seg5p_string_1', mode='w')
ssms_2.to_hdf('resources/ssm_store_pub1.hdf', key='mdb_127_en_seg5p_string_2', mode='a')  # Note the mode is set to append
songs_df[['id', 'borders']].to_hdf('resources/borders_pub1.hdf', key='mdb_127_en_seg5p', mode='w')

with pd.HDFStore('resources/mldb_watanabe_5plus_seg_english.hdf') as store:
    songs = store['mldb_watanabe_5plus_seg_english']

with pd.HDFStore('resources/ssm_store_pub1.hdf') as store:
    df1 = store['mdb_127_en_seg5p_string_1']
    df2 = store['mdb_127_en_seg5p_string_2']
    ssms_string = pd.concat([df1, df2], ignore_index=True)
    ssms_string.set_index(['id'], inplace=True)

with pd.HDFStore('resources/borders_pub1.hdf') as store:
    borders = store['mdb_127_en_seg5p']
    borders.set_index(['id'], inplace=True)

#select some song here
song = songs.iloc[12]

song_id = song.id
lyric = song.a_lyrics

#get borders and SSM from stores
segm_borders = borders.loc[song_id].borders
ssm_lines_string = ssms_string.loc[song_id].ssm

print('segment borders:', segm_borders, '\n')
# print(pretty_print_tree(tree_structure(normalize_lyric(lyric))))

#can show different encoding here, for now it's the same everywhere
ssm.draw_ssm_encodings_side_by_side(ssm_some_encoding=ssm_lines_string, ssm_other_encoding=ssm_lines_string, ssm_third_encoding=ssm_lines_string,\
                                    representation_some = 'string', representation_other = 'string', representation_third = 'string',\
                                    artist_name=song.a_name, song_name=song.a_song, genre_of_song='undef')
