
import os
import time
import base64
from io import BytesIO
import streamlit as st
from PIL import Image

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Check your secrets.")
    st.stop()

st.set_page_config(
    page_title="Chat With Documents",
    page_icon="📚",
    layout="wide"
)

# Embedded chat logos cropped from the supplied reference image.
BOT_LOGO = Image.open(BytesIO(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAC8AAAAuCAYAAAC4e0AJAAASF0lEQVR4nI2Z649e11XGf2vvfW7vZS4ej8eOHdtJG9KapEkd05aUkLaCUpUiSiVuEgjxgYL4G/gP4DsS8BEkkBCIS6DQilQlTZumSXFubi6+JHZ8ie0Zz8x7O+fsvRYfzpmpK2HUI+33vBrNe85aaz/rWWs9W5aXj5ppIAslddOSFzmtLhAXWdQTllfGxFpBBUQwM0wVM0DAiQMBM+Nel1MjKHhTwBBRQEEUQzExFEFFUPH9CiQJKA7F4Szh0oIghpnhnCeYOdpGqeczNg4dYvP2B4jUjJYyNpZG3Lh2mZUSxiVUVUmWBUBQVRDwzgOgmu5pvJgSLOEsgew52d1VBENoVDDJMF+yO2/Y2l1QqycrhkgoiMloU0LVSKrkzhFcyBiEklS3ECPSzhkNlLR9i/lU+dj98ItnjnDq+JCTJ09y9Oh9LC0t4ZxHVe8yWu5pPKKYREz0rv8VwKGEbknJ9rTlwuVb/PCdK5x94wL/89olLl9dIB5WD4zYip7kcsyMlCIyXD5u1jqCKT415EzJ2OXBEyOOHCr4g9/9Ao/fD2vFDoTQvTNGUtuSUmeM9x7n3D1tN1HURUwSsuekOcBjBJQMpaTVjChDQrnKrM145Y2LfO3r3+K/n7/EhQ+ApQKpDgAwmUwJ5gzxireWlYHgmjmFwK9/6TRPP/kwDx6tGNoWtECck5IigM8CflB0DqlCjPcOPIbb3xjrEaN0iWPdd3FUwWFpzmR7TtCCx08eZO2Xf57VEPn7r1/hwqxGXM1oPMI5CBKgzANZbPAyZ3VJ+dXPf4zPPfkApx6quHnldULhyXuM+syBc4Ch9ZRmp8aAPMvvDRsM6VISwaBP3G4JmIDugOSIryglUORD8nHJoZV1hvknWD5ylD/725e4tjtnMKzIsowQtWa6mJLHHTIf+ZknH+J3fusXWClvQX2R5cEueTaG5NG2JTYJEemgEjxlPujtuzfbgMckx3BgCSHCj63Up0ELomQSQBcs5rdYtMr6IcdnP3eK59+Df/rG68znC0IIBO+hqHLCInL/Onz5C2dYGzW0u9chTMn9nLZWRCpCluPyottmTaARkgEe6DEvPQzQfWybOFocKr4nPvDW7cY+bKwBnzoYmhGT4H3OUhFQM2ZhweeffoxvffcNtncmFKNVXFBFo7BaCl/81FEeP+7w8+sMvIAWJC3RXGjzltqUthGsVmhS5wAJLIM4Bh2ClZ1Dad755EtkoWRtTd5OyG1OsDnSziEmoAQdg1VAjqknaUAY4tMKrhkQ2sCq1DxxIvIrnzrMeqHEnRnOJcPUsVQKH39wxHq5Q6G7FM4gBbCSiNG4hiSC8wWSFZAFcALWQpugdRBdV8xMMW3AEkgGvkCc4KwFXQANBMAH0Ayrc1LKUcl76vRgOS5VEAuIgSE1R4fbnPnwgLFr8dEIACLCYDDg+IkTEAJOE1iElHDOk1PgVfEKTmuQBmwBVkOqIRMoPU1qkNwR8ow6VixmDY45g2JMSAZqHdQ0dagSwFdIVeGlq9y4FicRkQbcvEtmyzAyiiLngQdPAGdx3hEQEIQ8zzly5DDoNojrCE4NEY9P4DCcgVgLdIlFECgzjERt28Tc0ZiRksPnA1w1pI0l27UwUCiDQ5yCSAf11FOsRAggEhBru3ogTbezWoAVGAHnHCeOH0ON3vg97gUGgyGpvonPXfcCAPO4OgAZBA/OOlxr6oLihBqjxiAbMV0Y713d4YdvX+fq9R0OHryfT3zsFCeWDeempBhwZJRF3uVESqQ4x2NdMEz6DiJ2yS8eqDATUqxZWRmRFGTfeOs+zJQYIz4E8B2cUAc6Aq36B7agQjRPmxxJA21QUgG1Dnjlnff52jfO8cL3r3D1Btx3bM6bFxJf/cpjDNYqTJS6ARcDuXeQtficjvtT7LAkdLvh2i7q4lCElBaI9LTKnvECIo7YRorgO290rw+JkE0g7oDOwSU0rIJfwahoLae2BsPYvNPy8stv8+x/XeHWJgzGI27dbvmXZ77D048vszbeYHm8RktFrYa2iug2RV53UdamMyaV4HrjMUy6/imIMq+n+KxLm7BXXEIIqCq+qiAtUNWukEoLbgoyAZljuYdswKKJXL+9xeUbC7bmm8Ss5uZt5Y1zV6gXUBZLCCvsTmZkwLPPfYdmepjl8UFiA+vLa9x38ACjgYCfUTjAJdDQ02YN0nWhJooIFJnj9u6MrBB0IQS7C/PiBLwDdT/qz03AechKINDiUT3A2+82fOu5c7x67jqbu3dogfkC7myBTSD3Smx2KKSmLOB7393m6qVtYvMmzuC+9ZKnP/0UnzizxHBl0Blbz7rc0mFHuX7v/d2OBO8QSV1ZE0cQOgZLKRF8gLoGL32XuAedAOIwhHl0OL/Bm+fP841vXmBrO2djY51RMaNEOXJ8RH2oomnH5IMKKRrmi4uUxS4hdHHQBN95ccFi8TqHjj7C0kEQHwlujqQlYNxXbofqHMlaVCNER1ms07SGyT7b/H+XgbW0zQIZrpIPVpg0nqtbd9iaN5x8+MN88Zc+yX2HA2UAry3OFaSY43JhZ36TvPooRaFMZhNcWOLSu9v83V8/x6VrNbd3c2apAKsZOiV3AWzQ1ZkYSQl8nhCJOBxi2rd0/icwXgyTmuhrQuFpcGzOJ2zHCU2AwZpx8uFlThwOVGGKi7s422E4rNiZbtIyJStWaOp1VE6QQqC1G4TBCje3jFuTnBgOI4XStDexpBS1dhQpAe8yXFCkSbj9fugnNR5HTAVFtcq0WeG181u8fuEq569sMkswSzUJQ5xjsrtFiB+wNAJtN8ncjOEgo25aNjcXrK6Pqf0OvpyjQYh+yEuvXGWwpDzxiHF0bYT5jKQN3iUQwYUMSFgyJBiC9ZH/CWBjlqFpiLcNzl+Y8Mwzb/P8q+/y3i1jPofkS9QNuHFjlx/+4GWydI0nHn+A9YMlowMrQOD1197n5RcvMF67wSNPrzNvr7Kz2OTqrZZn/vMSL5+F3/jyBl/50kHuH1R47btUTViMSA2iHswQ9vrQnyjynhCW2N31vPDd8zz37Uu8dR3qHPICJM+YN8oPX32dv/nLtygVvvCLr/BHX/0Mk9uRGHP+6s/P8s4bEIZv8iePf5ZytGC8WlCOx4iVfLC5w7e+/R4njt3hwJlj5CEBEeqGpmnwlpOVo47G6EYbw3PvwXP/ErwviK3jypVNdnagGMBwNScbQCsNrRlZuUxRQsi6uWSw+iBtvYrjATI8selovF7cYbK4wrTeZNYsaHXAbHGAS5fhyrVd6qg0cReLEwiJ4B2WPEjZN2kdlA1HMPx+Ipjoj41ohkfxpGZBVQ0ZDD3DEezMYTpvEAGhpciER049zO//3s/S3rnEZz59mna7ZVAdJKYlvvqHf8y3v/k6Rz+8wsmTDe9sbpE7GBdCu5hSOVgbwfrKEVbGB2B3F6NGBi3eC+1uTbYw8AGxjCSe5CCYFeSaCBYxF1GLOOlmzSg5iRxcg9htnnrqAd6+9g43X4FSIPOQN3OKNOX+jVXu/9nHGNiHGFaeWNckZvhywvGHSzYeeJy5buOLTbK4wqrLySfXKWkoAnzuiQ0+8dAp2q2asRtDtgvxfUw9ebUGLkAqcFrROk8dUhf5oBFniknCSOw124rHEMq8BZ949CMr/NqXTvPAR27y/XNXeeetRKaJ3MHOndtceON/mNy6QpYHxAcWJsxVqFNDCC1r60M+/slHoB6j04aDVcETp4/x2MfWePzUYQ4vD6h8DlRdVysLzOV0Q7qH5BECKg5zibAnu3XF6C7hyBRHwpnC9g3wieXqCD/36Id46MOPkPQ7vH/xbVwoaN2Adze3+Pfvvc0br97GPLgKZgoTBa9wQOCJnx7y8KNPEtOCuoVimPHo6Yf5zGcfYG0pUcoWPhm0BpqjaRlLS6BHwQKwi0jsB3glyH6k+/6mEx4BxVnEYd3kRIR2wjgr8MMlRoXHEsQYmTdKIyVWrcHSbZILNNmQmQZSGFAFIZ9vEcIGdTMkphIJAQmR1YMV5dhQN0HdjDY5vPY84ocgK2AHuooru5i0nV2mBGTP+C6Lsb1BxHBETAQZrID31G3JlIIoGS3dbFKFxFLWcmh9heVfOMPiUw/h84KIZ5GEGAqcBGgia6sDlg8MubWjFENj586CUEYk1LRxF88U1U58dX6KC3PEpih3sKSI1GARTyJoInSayV7Jdb0D0rNOi+GJZKgWTChJ2ZBGAuoczkEzvcPtaxcZtKss+5ojqxlZ5kiqqHiSS0wapS4H5IMBl69c5vKVy8zmiaiQbIIPyzhakAXiBcEhYYGEGtMZKe2AGsE1mEa8QbBEEFEM6+TlXvR0pC5Jesw7NSCQITQJJjGyXi4xyuDC+QX/+MwLFFmBxoQDWhPq5JBigGUlzdQoZgVroxHzdJlp/QHb12HjUMkwq/DWkrmWjAWZFwJuf2JLSRB1XTqKQ6VTiPYjb2KYCEbAyDp8Q3e3PoEkx5HI6zlVNuTRB3+KNz96jedefJfnX7xBu5dnvmBhGbutQ7MWikgx8xyewAEW1NwkLyYcWBvz6TOf4viR+5B0G2yOd/MuxzDQGkMhCUIn5Nqe8XuRRwxzQiKQKMAPiLtTwqDq9EOfYG/iR0nUhAA/dbLgK7/0MA8dz7hzZwISEPEEn6HiqSMkCRBynCoDEsQJjRo+H7C+dh+nH1vj8EEjS4prCzK/hsgQ2gxshrgMIcehSAg0FLhs3GtdiWBYJ3DhmTUwCBk5eScK1doNwoXrNBrRDptERkXOmVMVZ049ijfDqyGWOqFpXwnur2xGXd6myUE5AOTQHkTSLkGNTA2vI0SXQQbQuE6FyzIcjkRCgzFXz6SfvzNxBDVoVZi3wtWbEwZrnuHSBrQ3wWdgLeo8SfyexouRMNpepuipVQxPQqxvMX5MCTZCXMJs1EHUPGiFqCOoEnSPooFYd8qbE6AAcQiBVjPmWnLu4gf4AIUUBJxHQsb2tOati9f4yLGPspjfphCPDCpUF9TOEyX0bERHrdbJFGKCw0hmODMcircu0WXPAS3xzSGwEusPGaATDIK13bOMXrhN4LL+IMN3Pb0EzA0gLPPya8/TtN2BRcAFoGBrZ5uXzr7NF586zeZOZGO1wjMFMrJYEizvi1j/adJ3d3tHNIJgiNEZzt070LGHE+13rdsNkT143aXTeweZ6+g71agYljlqCrbrwNlzV5nMwXxDUBMMxzwKL519k7OvX+T0hw7hyzmz+XXKLBDaAmLFj+TovTbC9YrWXfe7KnRnfAJXQ7aL+Gbf0c7ZTofEsv5vvleWFXRBrVOkAFcJu9Oc77065dz5W8TuIAWXTEh48mrMlWvwta//N+YqkgXq1LWh3bv65JXYtQvSdJKE1N19f7X/x0p98QudgQhIwu7+LW2nOGvs5AzLkLBCVm0QWeL9Gwv+9T9f4NqtmpBVhCwjKF1F83lFswPfffES//JvX+fJJ45y+L6S5GZYiHgxOi1+r53oZV4RTAJJPNYPCQ7tits++7hetx92QXAtJjNwDUbTQUyyXi/yiDlUhiAD5rXjrffO8x/PvsmLP7hOosC5HDXDuT5RmjahAS7dhD/9ixf45tnr3OEkt+oNFjYkie3BvN8J+RFKelXLJPURTago6rSrKOKADCPHCGAe6Y3sAmDdCJYXJFeiMqSxMbuLARffbfivZ8/z9/9wjZ0JlIMDNEmYNy0yXN2wlAxnLaMSMknMdnYocvj85z/Jb3/55zh9eJtD7lovTSdSbHs50OOzDETQpH3y0qe07S/MY2SYOZCISNtJ2PuV3JE0p2k8KmPK0QbTuuSFl97hn//tWV78wR1uLGBWDandqFeRBVlZO2LOORaLGWban3BD09aEkLFWLfjNp9b4+HHh+PHjHDt2jNWVFULIiCnSNA3QaZ33uqwfJkxAiHetruIYQlGtMFvAG29d5exr7/LKufc599Z1rt6AaOBGK2ynjNZlPwrQeOWQxRTx3vUzqZHlGd57mqamnmxxaJRYziHLIXjBDFRt/wBQ9gjknlc37XffUr8jP/6bRQNRAR9YpJzpwlOnAnMDVCpUE07qzuG9p46WD5lZoqoqDGU2m9G2DUWRU5YlToTZbAYY3vlOs+952XmH9x56rfP/i3zX9AmO7ihTpD/C3KvCEvBZjvkB89YxXRiQQzEGV0IzZWDbBGv2n/u/n52K/USWZKAAAAAASUVORK5CYII="))).convert("RGBA")
USER_LOGO = Image.open(BytesIO(base64.b64decode("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"))).convert("RGBA")


# =====================================================
# SPACE THEME - UI ONLY (BACKEND UNCHANGED)
# =====================================================

st.markdown("""
<style>
:root {
    color-scheme: dark;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: transparent !important;
    color: #F5F5F5 !important;
}

.stApp {
    background: #02040A !important;
}

/* Full-screen animated space background */
#space-background {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
    background:
        radial-gradient(ellipse at 50% 45%, rgba(17, 24, 50, 0.28), transparent 58%),
        #02040A !important;
}

/* Dense foreground stars: each radial-gradient is a star, repeated as a tile. */
#space-background .stars,
#space-background .stars-far {
    position: absolute !important;
    display: block !important;
    inset: -100% !important;
    width: 300% !important;
    height: 300% !important;
    max-width: none !important;
    pointer-events: none !important;
    background-repeat: repeat !important;
    transform-origin: 50% 50% !important;
    will-change: transform !important;
}

#space-background .stars {
    opacity: 0.95 !important;
    background-image:
        radial-gradient(2px 2px at 12px 18px, #fff 98%, transparent 100%),
        radial-gradient(1.5px 1.5px at 62px 91px, #fff 98%, transparent 100%),
        radial-gradient(1px 1px at 123px 37px, #eaf2ff 98%, transparent 100%),
        radial-gradient(2px 2px at 171px 143px, #fff 98%, transparent 100%),
        radial-gradient(1px 1px at 215px 72px, #fff 98%, transparent 100%),
        radial-gradient(1.5px 1.5px at 248px 188px, #dbeafe 98%, transparent 100%),
        radial-gradient(1px 1px at 34px 226px, #fff 98%, transparent 100%),
        radial-gradient(2px 2px at 194px 239px, #fff 98%, transparent 100%);
    background-size: 260px 260px !important;
    animation: stars-approach 70s linear infinite !important;
}

#space-background .stars-far {
    opacity: 0.65 !important;
    background-image:
        radial-gradient(1px 1px at 25px 45px, #dbeafe 98%, transparent 100%),
        radial-gradient(1px 1px at 87px 154px, #fff 98%, transparent 100%),
        radial-gradient(1.5px 1.5px at 151px 22px, #fff 98%, transparent 100%),
        radial-gradient(1px 1px at 205px 112px, #dbeafe 98%, transparent 100%),
        radial-gradient(1px 1px at 236px 218px, #fff 98%, transparent 100%),
        radial-gradient(1px 1px at 116px 246px, #fff 98%, transparent 100%);
    background-size: 270px 270px !important;
    animation: stars-approach 13s linear infinite !important;
}

@keyframes stars-approach {
    0%   { transform: scale(0.35) translate3d(0, 0, 0); }
    100% { transform: scale(1.35) translate3d(0, 0, 0); }
}

/* Keep Streamlit content above the animated background */
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stSidebar"] > div {
    position: relative;
    z-index: 1;
}

/* ChatGPT-like black sidebar, separated by a fine white border */
section[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid rgba(255,255,255,0.42) !important;
}

section[data-testid="stSidebar"] > div {
    background: #050505 !important;
}

section[data-testid="stSidebar"] * {
    color: #F3F4F6;
}

/* Main area stays black; stars remain visible around the content */
[data-testid="stMain"] {
    background: rgba(0,0,0,0.35) !important;
}

/* Question and answer panels: same black, thin white outline */
[data-testid="stChatMessage"] {
    background: rgba(5,5,5,0.94) !important;
    border: 1px solid rgba(255,255,255,0.38) !important;
    border-radius: 12px !important;
    color: #F5F5F5 !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 0 14px rgba(255,255,255,0.035);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] strong {
    color: #F5F5F5 !important;
}

/* Chat input panel */
[data-testid="stChatInput"] {
    background: #050505 !important;
    border: 1px solid rgba(255,255,255,0.48) !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
    background: #050505 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #A1A1AA !important;
}

/* Other input controls */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
[data-testid="stFileUploader"] section {
    background: #080808 !important;
    color: #FFFFFF !important;
    border-color: rgba(255,255,255,0.35) !important;
}

/* Buttons */
.stButton button {
    background: #080808 !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.48) !important;
    border-radius: 9px !important;
}

.stButton button:hover {
    background: #202020 !important;
    border-color: #FFFFFF !important;
}

/* Text and headings */
.stApp h1, .stApp h2, .stApp h3,
.stApp p, .stApp label,
.stApp [data-testid="stCaptionContainer"] {
    color: #F5F5F5 !important;
}

/* Expanders / sources */
[data-testid="stExpander"] {
    background: #050505 !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] p {
    color: #F5F5F5 !important;
}

hr {
    border-color: rgba(255,255,255,0.25) !important;
}

/* Reduce motion for users who prefer it */
@media (prefers-reduced-motion: reduce) {
    #space-background .stars,
    #space-background .stars-far,

}
</style>

<div id="space-background" aria-hidden="true">
    <div class="stars"></div>
    <div class="stars-far"></div>
</div>
""", unsafe_allow_html=True)


st.title("📚 Chat With Your Documents")
st.caption("Upload PDF documents and ask questions using Google Gemini.")


# =====================================================
# 2. INITIALIZE GEMINI EMBEDDINGS
# =====================================================

@st.cache_resource
def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )


# =====================================================
# 3. INITIALIZE GEMINI LLM
# =====================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        max_output_tokens=512,
        google_api_key=GOOGLE_API_KEY
    )


# =====================================================
# 4. EXTRACT TEXT FROM PDF
# =====================================================

def get_pdf_documents(pdf_files):

    documents = []

    for pdf_file in pdf_files:

        temp_path = f"temp_{pdf_file.name}"

        try:

            with open(temp_path, "wb") as file:
                file.write(pdf_file.getvalue())

            loader = PyPDFLoader(temp_path)

            pdf_pages = loader.load()

            for page in pdf_pages:

                page.metadata["source"] = pdf_file.name

            documents.extend(pdf_pages)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

    return documents


# =====================================================
# 5. SPLIT TEXT INTO CHUNKS
# =====================================================

def get_text_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# =====================================================
# 6. CREATE FAISS VECTOR DATABASE
# =====================================================

def create_vector_store(chunks):

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")

    return vector_store


# =====================================================
# 7. EXTRACT CLEAN GEMINI RESPONSE
# =====================================================

def extract_response_text(content):

    if isinstance(content, str):

        return content.strip()

    elif isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    text_parts.append(block.get("text", ""))

            elif isinstance(block, str):

                text_parts.append(block)

        return "\n".join(text_parts).strip()

    return ""


# =====================================================
# 8. GENERATE ANSWER USING GEMINI
# =====================================================

def generate_answer(question, vector_store):

    start_time = time.perf_counter()

    # Retrieve relevant chunks
    relevant_docs = vector_store.similarity_search(
        question,
        k=7
    )

    retrieval_time = time.perf_counter()

    if not relevant_docs:

        return (
            "Answer is not available in the uploaded documents.",
            []
        )

    # Prepare context and sources
    context_parts = []

    sources = []

    for doc in relevant_docs:

        source = doc.metadata.get("source", "Unknown")

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"""
            Document: {source}
            Page: {page}

            Content:
            {doc.page_content}
            """
        )

        sources.append({
            "file": source,
            "page": page
        })

    context = "\n\n".join(context_parts)

    # Remove duplicate sources
    unique_sources = []

    for source in sources:

        if source not in unique_sources:

            unique_sources.append(source)

    # Prepare prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent document question-answering assistant.

    Your task is to answer questions using ONLY the provided context.

    IMPORTANT RULES:

    1. Use only the information available in the context.
    2. Do not use outside knowledge.
    3. Do not invent facts or information.
    4. If the answer is not present in the context,
       reply exactly:
       "Answer is not available in the uploaded documents."
    5. Provide clear and detailed answers.
    6. Use bullet points when appropriate.
    7. Format mathematical equations and chemical reactions
       in readable notation.
    8. Never display internal metadata, signatures,
       JSON, or technical response information.

    CONTEXT:

    {context}

    QUESTION:

    {question}

    ANSWER:
    """)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # Generate answer
    llm = get_llm()

    response = llm.invoke(formatted_prompt)

    # Extract only readable text
    answer = extract_response_text(response.content)

    if not answer:

        answer = "Unable to generate an answer. Please try again."

    end_time = time.perf_counter()

    # Performance information
    print("\n========== PERFORMANCE ==========")

    print(
        f"Retrieval time: "
        f"{retrieval_time - start_time:.2f} seconds"
    )

    print(
        f"Gemini generation time: "
        f"{end_time - retrieval_time:.2f} seconds"
    )

    print(
        f"Total time: "
        f"{end_time - start_time:.2f} seconds"
    )

    print("=================================\n")

    return answer, unique_sources


# =====================================================
# 9. SESSION STATE
# =====================================================

if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =====================================================
# 10. SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📁 Document Manager")

    pdf_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Process Documents",
        use_container_width=True
    ):

        if not pdf_files:

            st.warning("Please upload PDF documents first.")

        else:

            try:

                # Extract text
                with st.spinner("Extracting PDF text..."):

                    documents = get_pdf_documents(pdf_files)

                if not documents or not any(
                    doc.page_content.strip()
                    for doc in documents
                ):

                    st.error(
                        "No readable text found in the uploaded PDFs. "
                        "Scanned PDFs may require OCR."
                    )

                else:

                    # Split text
                    with st.spinner("Splitting documents into chunks..."):

                        chunks = get_text_chunks(documents)

                    # Create embeddings and FAISS
                    with st.spinner(
                        "Creating embeddings and FAISS database..."
                    ):

                        vector_store = create_vector_store(chunks)

                    st.session_state.vector_store = vector_store

                    st.session_state.chat_history = []

                    st.success("Documents processed successfully!")

                    st.write(f"Total documents: {len(pdf_files)}")

                    st.write(f"Total pages: {len(documents)}")

                    st.write(f"Total chunks: {len(chunks)}")

            except Exception as e:

                st.error(f"Document processing failed: {str(e)}")


    st.divider()

    if st.button("Clear Chat", use_container_width=True):

        st.session_state.chat_history = []

        st.rerun()


    if st.button("Remove Documents", use_container_width=True):

        st.session_state.vector_store = None

        st.session_state.chat_history = []

        st.rerun()


# =====================================================
# 11. DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.chat_history:

    # Render the logo as a normal Streamlit image (not a chat avatar).
    # This works consistently across Streamlit versions.
    logo = USER_LOGO if message["role"] == "user" else BOT_LOGO

    with st.chat_message(message["role"]):
        logo_col, message_col = st.columns([0.75, 12], gap="small")

        with logo_col:
            st.image(logo, width=42)

        with message_col:
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):

                with st.expander("📖 View Sources"):

                    for source in message["sources"]:

                        st.write(
                            f"📄 {source['file']} "
                            f"- Page {source['page']}"
                        )


# =====================================================
# 12. CHAT INPUT
# =====================================================

if st.session_state.vector_store is None:

    st.info(
        "👈 Upload your PDF documents and click "
        "'Process Documents' to start chatting."
    )

else:

    user_question = st.chat_input(
        "Ask a question about your documents..."
    )

    if user_question:

        # Display question with an explicit logo image.
        with st.chat_message("user"):
            logo_col, message_col = st.columns([0.75, 12], gap="small")
            with logo_col:
                st.image(USER_LOGO, width=42)
            with message_col:
                st.markdown(user_question)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Generate response with the DocuMind logo shown explicitly.
        with st.chat_message("assistant"):
            logo_col, message_col = st.columns([0.75, 12], gap="small")
            with logo_col:
                st.image(BOT_LOGO, width=42)
            with message_col:

                try:

                    with st.spinner("Gemini is analyzing your documents..."):

                        answer, sources = generate_answer(
                            user_question,
                            st.session_state.vector_store
                        )

                    st.markdown(answer)

                    # Display sources
                    if sources:

                        with st.expander("📖 View Sources"):

                            for source in sources:

                                st.write(
                                    f"📄 {source['file']} "
                                    f"- Page {source['page']}"
                                )

                    # Save answer
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(f"Error generating answer: {str(e)}")
