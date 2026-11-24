"""MixBhasha v5 research demo. Run: streamlit run app.py"""
from pathlib import Path
import sys, os, json
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import pandas as pd
import streamlit as st
import mixbhasha_core as core

DIR=Path(__file__).resolve().parent
st.set_page_config(page_title='MixBhasha v5',page_icon='🗣️',layout='wide')
st.title('🗣️ MixBhasha v5')
st.caption('Hinglish code-mixed sentiment and cyberbullying analysis · research prototype')
with st.sidebar:
    st.header('Models')
    root=Path(st.text_input('Trained experiment root',os.getenv('MIXBHASHA_ROOT',str(DIR/'run_outputs')))).expanduser()
    st.info('No automatic punishment, bans, or moderation decisions. Human review is essential.')
lockfile=root/'results'/'locked_validation_winners.json'
if not lockfile.exists():
    st.warning('No locked models found. Train and finalize them in the notebook first.')
    st.code(str(lockfile));st.stop()
locks=json.loads(lockfile.read_text())
@st.cache_resource(show_spinner='Loading checkpoint…')
def cached(root_string,task,name):
    return core.load_model(Path(root_string),task,name)
try:
    cases=json.loads((DIR/'examples'/'challenge_cases.json').read_text())
except Exception:
    cases=[['sarcasm','Wah kya service hai, 3 ghante wait karwaya, amazing 🙃'],
           ['mixed_script','कल class hai but I am not ready'],
           ['neutral','Kal 9 baje meeting hai; link bhej dena']]
options={'Custom input':''}
options.update(dict(cases))
selection=st.selectbox('Example (unlabeled, not a benchmark)',list(options))
text=st.text_area('Enter Hinglish / English / Hindi',options[selection] or 'Yaar movie buri nahi thi, but ending weak thi',height=110)
if st.button('Analyze',type='primary',disabled=not text.strip()):
    tasks=[k for k in ('sentiment_raw','sentiment','cyberbullying_binary','cyberbullying7','offense_hasoc','hate') if k in locks]
    for task in tasks:
        st.subheader({'sentiment_raw':'Sentiment (raw team-mirror, 3-class)','sentiment':'Sentiment (3-class)','cyberbullying_binary':'Cyberbullying-category flag',
                      'cyberbullying7':'Cyberbullying annotated category','offense_hasoc':'HASOC HOF / NOT',
                      'hate':'Separate hate classification'}[task])
        with st.container(border=True):
            try:
                model_name=locks[task]['model']
                loaded=cached(str(root),task,model_name)
                pred=core.predict_text(root,task,text,loaded=loaded)
                l,r=st.columns([3,1])
                with l:
                    st.markdown(f'**Predicted:** `{pred["label"]}`')
                    st.bar_chart(pd.DataFrame({'class':list(pred['probabilities']),
                              'score':list(pred['probabilities'].values())}).set_index('class'))
                with r:
                    st.metric('Review required','YES' if pred['review_required'] else 'No')
                    st.metric('Raw top score',f'{pred["uncalibrated_max_probability"]:.1%}')
                    st.caption('NOT a calibrated probability of correctness')
                st.caption(f'Model: {model_name}; score display is uncalibrated; review cutoff is validation-derived.')
            except Exception as e:
                st.error(f'{type(e).__name__}: {e}')
    if 'cyberbullying7' not in tasks and 'cyberbullying_binary' not in tasks and 'offense_hasoc' not in tasks:
        st.warning('No cyberbullying / offensive-language model is locked for this experiment.')
st.divider()
st.markdown('''**Interpretation limits:** MC-Hinglish1.0 seven categories are mutually exclusive annotation labels, not a complete inventory of harmful language. Its binary task groups six annotated categories against `not_cyberbullying`; this is **not** the same as HASOC `HOF/NOT` or PRISM hate/non-hate. Sarcasm, negation, mixed scripts, and quotation can still cause failure. The processed SentiMix dev set was seen in v4; the raw team-mirror test is not organizer-verified official gold.''')
st.caption('For research/demo only. No model weights are supplied with this package.')
