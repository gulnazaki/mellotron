import re
import numpy as np
import torch
import torch.nn.functional as F
from text import text_to_sequence, get_arpabet, cmudict
from string import capwords


CMUDICT_PATH = "data/cmu_dictionary"
CMUDICT = cmudict.CMUDict(CMUDICT_PATH)
PHONEME2GRAPHEME = {
    'AA': ['a', 'o', 'ah'],
    'AE': ['a', 'e'],
    'AH': ['u', 'e', 'a', 'h', 'o', 'ou'],
    'AO': ['o', 'u', 'au', 'a'],
    'AW': ['ou', 'ow'],
    'AX': ['a'],
    'AXR': ['er'],
    'AY': ['i'],
    'EH': ['e', 'ae', 'ea'],
    'EY': ['a', 'ai', 'ei', 'e', 'y'],
    'IH': ['i', 'e', 'y', 'ea'],
    'IX': ['e', 'i'],
    'IY': ['ea', 'ey', 'y', 'i', 'e', 'ie'],
    'OW': ['oa', 'o', 'oh'],
    'OY': ['oy'],
    'UH': ['oo', 'u'],
    'UW': ['oo', 'u', 'o', 'ou', 'ew'],
    'UX': ['u'],
    'B': ['b'],
    'CH': ['ch', 'tch', 't'],
    'D': ['d', 'e', 'de', 'ed'],
    'DH': ['th'],
    'DX': ['tt'],
    'EL': ['le'],
    'EM': ['m'],
    'EN': ['on'],
    'ER': ['i', 'ir', 'er', 'r', 'ere', 'ar'],
    'F': ['f', 'fe'],
    'G': ['g'],
    'HH': ['h'],
    'JH': ['j'],
    'K': ['k', 'c', 'ck', 'ke'],
    'KS': ['x'],
    'L': ['ll', 'l'],
    'M': ['m', 'me'],
    'N': ['n', 'gn', 'ne', 'kn'],
    'NG': ['ng', 'n'],
    'NX': ['nn'],
    'P': ['p'],
    'Q': ['-'],
    'R': ['wr', 'r', 're', 'er'],
    'S': ['s', 'c', 'ce'],
    'SH': ['sh'],
    'T': ['t', 'ght', 'ed'],
    'TH': ['th'],
    'V': ['v', 'f', 'e', 've'],
    'W': ['w', 'wh'],
    'WH': ['wh'],
    'Y': ['y', 'j'],
    'Z': ['z', 's', 'se'],
    'ZH': ['s']
}

########################
#  CONSONANT DURATION  #
########################
PHONEMEDURATION = {
    'B': 0.05,
    'CH': 0.1,
    'D': 0.075,
    'DH': 0.05,
    'DX': 0.05,
    'EL': 0.05,
    'EM': 0.05,
    'EN': 0.05,
    'F': 0.1,
    'G': 0.05,
    'HH': 0.05,
    'JH': 0.05,
    'K': 0.05,
    'L': 0.05,
    'M': 0.15,
    'N': 0.15,
    'NG': 0.15,
    'NX': 0.05,
    'P': 0.05,
    'Q': 0.075,
    'R': 0.05,
    'S': 0.1,
    'SH': 0.05,
    'T': 0.075,
    'TH': 0.1,
    'V': 0.05,
    'Y': 0.05,
    'W': 0.05,
    'WH': 0.05,
    'Z': 0.05,
    'ZH': 0.05
}


def add_space_between_events(events, connect=False):
    new_events = []
    for i in range(1, len(events)):
        token_a, freq_a, start_time_a, end_time_a = events[i-1][-1]
        token_b, freq_b, start_time_b, end_time_b = events[i][0]

        if token_a in (' ', '') and len(events[i-1]) == 1:
            new_events.append(events[i-1])
        elif token_a not in (' ', '') and token_b not in (' ', ''):
            new_events.append(events[i-1])
            if connect:
                new_events.append([[' ', 0, end_time_a, start_time_b]])
            else:
                new_events.append([[' ', 0, end_time_a, end_time_a]])
        else:
            new_events.append(events[i-1])

    if new_events[-1][0][0] != ' ':
        new_events.append([[' ', 0, end_time_a, end_time_a]])
    new_events.append(events[-1])

    return new_events


def adjust_words(events):
    new_events = []
    for event in events:
        if len(event) == 1 and event[0][0] == ' ':
            new_events.append(event)
        else:
            for e in event:
                if e[0][0].isupper():
                    new_events.append([e])
                else:
                    new_events[-1].extend([e])
    return new_events


def adjust_extensions(events, phoneme_durations):
    if len(events) == 1:
        return events

    idx_last_vowel = None
    n_consonants_after_last_vowel = 0
    target_ids = np.arange(len(events))
    for i in range(len(events)):
        token = re.sub('[0-9{}]', '', events[i][0])
        if idx_last_vowel is None and token not in phoneme_durations:
            idx_last_vowel = i
            n_consonants_after_last_vowel = 0
        else:
            if token == '_' and not n_consonants_after_last_vowel:
                events[i][0] = events[idx_last_vowel][0]
            elif token == '_' and n_consonants_after_last_vowel:
                events[i][0] = events[idx_last_vowel][0]
                start = idx_last_vowel + 1
                target_ids[start:start+n_consonants_after_last_vowel] += 1
                target_ids[i] -= n_consonants_after_last_vowel
            elif token in phoneme_durations:
                n_consonants_after_last_vowel += 1
            else:
                n_consonants_after_last_vowel = 0
                idx_last_vowel = i

    new_events = [0] * len(events)
    for i in range(len(events)):
        new_events[target_ids[i]] = events[i]

    # adjust time of consonants that were repositioned
    for i in range(1, len(new_events)):
        if new_events[i][2] < new_events[i-1][2]:
            new_events[i][2] = new_events[i-1][2]
            new_events[i][3] = new_events[i-1][3]

    return new_events


def adjust_phoneme_lengths(events, phoneme_durations):
    t_init = events[0][2]
    t_end = events[0][3]
    consonant_lengths = {}
    vowel_lengths = {}
    for event in events:
        c = re.sub('[0-9{}]', '', event[0])
        if c in phoneme_durations:
            consonant_lengths[event[0]] = phoneme_durations[c]
        else:
            vowel_lengths[event[0]] = 0

    vowel_duration = (t_end - t_init - sum(consonant_lengths.values())) / len(vowel_lengths)
    vowel_lengths = {k : vowel_duration for k in vowel_lengths}

    time = t_init
    for i in range(len(events)):
        phoneme = events[i][0]
        if phoneme in vowel_lengths:
            phoneme_lengths = vowel_lengths
        else:
            phoneme_lengths = consonant_lengths
        
        events[i][2] = time
        if i < len(events) - 1:
            time += phoneme_lengths[phoneme]
            events[i][3] = time
        else:
            events[i][3] = t_end

    return events


def adjust_phonemes(events, phoneme_durations):
    if len(events) == 1:
        return events

    start = 0
    split_ids = []
    t_init = events[0][2]

    # get each substring group
    for i in range(1, len(events)):
        if events[i][2] != t_init:
            split_ids.append((start, i))
            start = i
            t_init = events[i][2]
    split_ids.append((start, len(events)))

    for (start, end) in split_ids:
        events[start:end] = adjust_phoneme_lengths(
            events[start:end], phoneme_durations)

    return events


def adjust_event(event, hop_length=256, sampling_rate=22050):
    tokens, freq, start_time, end_time = event

    if tokens == ' ':
        return [event] if freq == 0 else [['_', freq, start_time, end_time]]

    return [[token, freq, start_time, end_time] for token in tokens]


def track2events(track):
    events = []
    for e in track:
        events.extend(adjust_event(e))
    group_ids = [i for i in range(len(events))
                 if events[i][0] in [' '] or events[i][0].isupper()]

    events_grouped = []
    for i in range(1, len(group_ids)):
        start, end = group_ids[i-1], group_ids[i]
        events_grouped.append(events[start:end])

    if events[-1][0] != ' ':
        events_grouped.append(events[group_ids[-1]:])

    return events_grouped


def events2eventsarpabet(event):
    if event[0][0] == ' ':
        return event

    # get word and word arpabet
    letters = [e[0] for e in event]
    word = ''.join([l for l in letters if l not in('_', ' ')])
    word_arpabet = get_arpabet(word, CMUDICT)

    if word_arpabet[0] != '{':
        print("{} NOT IN CMUDICT\n".format(word))
        return []

    phonemes = word_arpabet.split()

    ip = 0
    ig = 0
    phoneme_events = []
    backtrack = []
    backtracking = False
    match = False
    spaces = 0
    while True:
        if ip == len(phonemes) and ig == len(letters):
            match = True
            break
        elif ig >= len(letters):
            if backtrack:
                ip, ig, spaces, valid_graphemes = backtrack.pop()
                phoneme_events = phoneme_events[:ip + spaces]
                backtracking = True
            else:
                break

        if letters[ig] == ' ':
            phoneme_events.append([' ', event[ig][1], event[ig][2], event[ig][3]])
            ig += 1
            spaces += 1
            continue

        if letters[ig] == "'":
            ig += 1
            continue

        if not backtracking:
            if ip < len(phonemes):
                possible_graphemes = PHONEME2GRAPHEME[re.sub('[0-9{}]', '', phonemes[ip])]
                valid_graphemes = [g for g in possible_graphemes if g == ''.join(letters[ig : ig + len(g)]).lower()]
            else:
                valid_graphemes = []

        if not valid_graphemes:
            if backtrack:
                ip, ig, spaces, valid_graphemes = backtrack.pop()
                phoneme_events = phoneme_events[:ip + spaces]
                backtracking = True
            else:
                break
        else:
            backtracking = False
            if len(valid_graphemes) > 1:
                backtrack.append((ip, ig, spaces, valid_graphemes[1:]))
            phoneme_events.append([phonemes[ip], event[ig][1], event[ig][2], event[ig][3]])
            ip += 1
            ig += len(valid_graphemes[0])

    if not match:
        print("NO PHONEME-GRAPHEME MATCH", word, word_arpabet, "\n", " ".join(p[0] for p in phoneme_events), "\n")
        return []
    return phoneme_events


def event2alignment(events, hop_length=256, sampling_rate=22050):
    frame_length = float(hop_length) / float(sampling_rate)

    n_frames = int(events[-1][-1][-1] / frame_length)
    n_tokens = np.sum([len(e) for e in events])
    alignment = np.zeros((n_tokens, n_frames))

    cur_event = -1
    for event in events:
        for i in range(len(event)):
            if len(event) == 1 or cur_event == -1 or event[i][0] != event[i-1][0]:
                cur_event += 1
            token, freq, start_time, end_time = event[i]
            alignment[cur_event, int(start_time/frame_length):int(end_time/frame_length)] = 1

    return alignment[:cur_event+1]


def event2f0(events, hop_length=256, sampling_rate=22050):
    frame_length = float(hop_length) / float(sampling_rate)
    n_frames = int(events[-1][-1][-1] / frame_length)
    f0s = np.zeros((1, n_frames))

    for event in events:
        for i in range(len(event)):
            token, freq, start_time, end_time = event[i]
            f0s[0, int(start_time/frame_length):int(end_time/frame_length)] = freq

    return f0s


def event2text(events, convert_stress):
    text_clean = ''
    for event in events:
        for i in range(len(event)):
            if i > 0 and event[i][0] == event[i-1][0]:
                continue
            if event[i][0] == ' ' and len(event) > 1:
                if text_clean[-1] != "}":
                    text_clean = text_clean[:-1] + '} {'
                else:
                    text_clean += ' {'
            else:
                if event[i][0][-1] in ('}', ' '):
                    text_clean += event[i][0]
                else:
                    text_clean += event[i][0] + ' '

    if convert_stress:
        text_clean = re.sub('[0-9]', '1', text_clean)

    text_encoded = text_to_sequence(text_clean, [], CMUDICT)
    return text_encoded, text_clean


def remove_excess_frames(alignment, f0s):
    excess_frames = np.sum(alignment.sum(0) == 0)
    alignment = alignment[:, :-excess_frames] if excess_frames > 0 else alignment
    f0s = f0s[:, :-excess_frames] if excess_frames > 0 else f0s
    return alignment, f0s


def split_multiword(events):
    new_events = []
    for event in events:
        words = event[0].split()
        if len(words) <= 1:
            new_events.append(event)
        else:
            durations = (event[3] - event[2]) / len(words)
            for i, word in enumerate(words):
                new_events.append([word, event[1], event[2] + i * durations, event[2] + (i + 1) * durations])
    return new_events


def get_data_from_text_events(text_events, ticks=True, midipath=None, tempo=120, resolution=220, phoneme_durations=None, convert_stress=False):
    def pitch_to_freq(pitch):
        return 440*(2**((pitch - 69)/12))

    if ticks:
        if midipath:
            from pretty_midi import PrettyMIDI
            midi = PrettyMIDI(midipath)
            to_time = midi.tick_to_time
        else:
            to_time = lambda t: t * 60/(tempo*resolution)
    else:
        to_time = lambda t: t

    if phoneme_durations is None:
        phoneme_durations = PHONEMEDURATION
    events = []
    time = 0
    notes_off = True
    start_of_word = True
    rest_start = -1
    for e in text_events:
        e_split = e.split('_')
        if '_' not in e:
            lyric = (capwords(e)) if start_of_word else (e.lower().rstrip())
            start_of_word = True if e[-1] == ' ' else False
        elif e_split[0] == 'ON':
            if rest_start >= 0:
                events.append([' ', 0, to_time(rest_start), to_time(time)])
                rest_start = -1
            freq = pitch_to_freq(int(e_split[1]))
            start = time
            notes_off = False
        elif e_split[0] == 'W':
            t = int(e_split[1])
            if notes_off and rest_start < 0:
                rest_start = time
            time += t
        elif e_split[0] == 'OFF':
            events.append([lyric, freq, to_time(start), to_time(time)])
            notes_off = True

    events = split_multiword(events)
    events = track2events(events)
    events = adjust_words(events)
    events_arpabet = [events2eventsarpabet(e) for e in events]
    events_arpabet = [e for e in events_arpabet if e]

    # make adjustments
    events_arpabet = [adjust_extensions(e, phoneme_durations)
                      for e in events_arpabet]
    events_arpabet = [adjust_phonemes(e, phoneme_durations)
                      for e in events_arpabet]
    events_arpabet = add_space_between_events(events_arpabet)

    # convert data to alignment, f0 and text encoded
    alignment = event2alignment(events_arpabet)
    f0s = event2f0(events_arpabet)
    alignment, f0s = remove_excess_frames(alignment, f0s)
    text_encoded, text_clean = event2text(events_arpabet, convert_stress)

    # convert data to torch
    alignment = torch.from_numpy(alignment).permute(1, 0)[:, None].float()
    f0s = torch.from_numpy(f0s)[None].float()
    text_encoded = torch.LongTensor(text_encoded)[None]

    return {'rhythm': alignment, 'pitch_contour': f0s, 'text_encoded': text_encoded}


if __name__ == "__main__":
    import argparse
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filepath", required=True)
    args = parser.parse_args()
    get_data_from_musicxml(args.filepath, 60)
