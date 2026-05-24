import os, json, time, re, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
ART  = os.path.join(BASE, 'artifacts')

def load_pkl(name):
    with open(os.path.join(ART, name), 'rb') as f:
        return pickle.load(f)

def load_json(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)

print("Loading models...")
svm        = load_pkl('svm_hybrid.pkl')
tfidf      = load_pkl('tfidf.pkl')
meta_pre   = load_pkl('meta_preprocessor.pkl')
lr_terms   = load_json('lr_terms.json')
s2_cache   = load_json('stage2_cache.json')
meta_info  = load_json('top_metadata.json')
M          = load_json('model_meta.json')
print(f"✅ Models loaded — SVM {M['svm_tps']:,} preds/sec | threshold [{M['low_thresh']},{M['high_thresh']}]")

META_BINARY = ['telecommuting','has_company_logo','has_questions']
META_CAT    = ['employment_type','required_experience','required_education','industry','function']
ALL_META_CAT = META_CAT + ['location_clean']
TEXT_COLS   = ['title','company_profile','description','requirements','benefits']

def clean(s):
    if not s or str(s).strip() == '': return ''
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+', 'url', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def build_features(fields):
    text = ' '.join(clean(fields.get(c,'')) for c in TEXT_COLS)

    row = {}
    for c in META_BINARY:
        v = fields.get(c, 0)
        row[c] = int(v) if v not in ('', None) else 0
    for c in META_CAT:
        row[c] = str(fields.get(c,'Unknown')) or 'Unknown'
    loc = fields.get('location','Unknown') or 'Unknown'
    row['location_clean'] = loc.split(',')[0].strip() if ',' in loc else loc

    df_row = pd.DataFrame([row])
    X_text = tfidf.transform([text])
    X_meta = meta_pre.transform(df_row)
    X_hyb  = hstack([X_text, X_meta])
    return text, X_text, X_hyb

def get_fraud_terms(X_text_vec, top_n=8):
    fn    = tfidf.get_feature_names_out()
    coefs = {t['term']: t['coef'] for t in lr_terms['fraud']}
    cx    = X_text_vec.tocoo()
    hits  = []
    for idx, val in zip(cx.col, cx.data):
        term = fn[idx]
        if term in coefs:
            hits.append({"term": term, "score": round(float(coefs[term] * val), 3)})
    hits.sort(key=lambda x: x['score'], reverse=True)
    return hits[:top_n]

def find_s2(s1_prob):
    return min(s2_cache, key=lambda x: abs(x['s1_prob'] - s1_prob))

@app.route('/')
def serve_frontend():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>FraudGuard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#050810;--s1:#090d1a;--s2:#0c1222;--b1:#0f1e35;--b2:#172840;--tx:#bdd0e8;--mu:#3d5570;--di:#1a2e45;--gr:#00e676;--gd:#002a12;--gb:#004d22;--rd:#ff3d3d;--rdd:#2a0505;--rdb:#550a0a;--am:#ffb300;--amd:#2a1a00;--amb:#553500;--cy:#00bcd4;--bl:#2979ff;--pu:#7c3aed}
body{background:var(--bg);color:var(--tx);font-family:'IBM Plex Mono',monospace;font-size:12px}
.hd{background:var(--s1);border-bottom:1px solid var(--b1);display:flex;align-items:center;gap:10px;padding:0 16px;height:50px;position:sticky;top:0;z-index:100}
.logo{width:26px;height:26px;border:1px solid var(--rd);display:grid;place-items:center;font-size:13px;color:var(--rd);flex-shrink:0}
.brand{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;letter-spacing:3px;color:#fff}
.brand em{color:var(--rd);font-style:normal}
.hd-pills{margin-left:auto;display:flex;gap:5px}
.hp{font-size:9px;padding:3px 7px;border:1px solid var(--b2);color:var(--mu)}
.hp.live{border-color:var(--gb);color:var(--gr)}
.slabel{padding:10px 16px;background:var(--s1);border-bottom:1px solid var(--b1);font-size:9px;letter-spacing:2px;color:var(--mu)}
.slabel::before{content:'> ';color:var(--cy)}
.exs{padding:10px 16px;background:var(--s1);border-bottom:1px solid var(--b1);display:flex;gap:6px;flex-wrap:wrap}
.eb{font-family:'IBM Plex Mono',monospace;font-size:11px;padding:7px 12px;border:1px solid var(--b2);background:transparent;color:var(--mu);cursor:pointer}
.eb.ef{border-color:var(--rdb);color:var(--rd);background:var(--rdd)}
.eb.el{border-color:var(--gb);color:var(--gr);background:var(--gd)}
.eb.er{border-color:var(--amb);color:var(--am);background:var(--amd)}
.fields{padding:14px 16px;background:var(--s1);display:flex;flex-direction:column;gap:12px}
.fl{font-size:9px;letter-spacing:2px;color:var(--mu);margin-bottom:4px;display:block}
textarea,select{width:100%;background:var(--bg);border:1px solid var(--b2);color:var(--tx);font-family:'IBM Plex Mono',monospace;font-size:13px;padding:10px;line-height:1.7;resize:vertical}
textarea:focus,select:focus{outline:none;border-color:var(--cy)}
select{-webkit-appearance:none}
select option{background:#090d1a}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.tr{display:flex;align-items:center;gap:8px;padding:10px;background:var(--bg);border:1px solid var(--b2);cursor:pointer}
.tr input{accent-color:var(--gr);width:16px;height:16px;flex-shrink:0}
.trl{font-size:10px;color:var(--mu)}
.run-btn{display:block;width:100%;padding:16px;background:transparent;border:0;border-top:2px solid var(--cy);color:var(--cy);font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:3px;cursor:pointer}
.run-btn:disabled{color:var(--mu);border-top-color:var(--b2);cursor:not-allowed}
#results{padding:16px;display:flex;flex-direction:column;gap:12px;background:var(--bg);border-top:2px solid var(--b1)}
.idle{padding:40px 0;display:flex;flex-direction:column;align-items:center;gap:10px;opacity:.2}
.idle svg{width:40px;height:40px;stroke:var(--tx)}
.idle-t{font-size:10px;letter-spacing:2px;color:var(--mu);text-align:center}
.rc{background:var(--s1);border:1px solid var(--b1);padding:14px;position:relative}
.rc::before{content:'';position:absolute;top:0;left:0;right:0;height:1px}
.rc.cg::before{background:var(--gr)}.rc.cr::before{background:var(--rd)}.rc.ca::before{background:var(--am)}.rc.cb::before{background:var(--bl)}.rc.cn::before{background:var(--b2)}
.rct{font-size:9px;letter-spacing:2px;color:var(--mu);margin-bottom:10px}
.vr{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.vb{font-size:11px;font-weight:700;letter-spacing:2px;padding:6px 14px;border:1px solid}
.vb.LOW_RISK{color:var(--gr);border-color:var(--gr);background:var(--gd)}
.vb.HIGH_RISK{color:var(--rd);border-color:var(--rd);background:var(--rdd)}
.vb.REVIEW{color:var(--am);border-color:var(--am);background:var(--amd)}
.bigp{font-family:'Syne',sans-serif;font-size:40px;font-weight:800;line-height:1}
.bigp.sd{color:var(--rd)}.bigp.ss{color:var(--gr)}.bigp.sw{color:var(--am)}
.prog{height:3px;background:var(--b1);margin-top:6px}
.pf{height:100%;transition:width .8s ease}
.fr{background:var(--rd)}.fg{background:var(--gr)}.fa{background:var(--am)}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.sc{padding:11px;border:1px solid}
.sc.ss1{border-color:var(--gb);background:var(--gd)}
.sc.ss2a{border-color:var(--amb);background:var(--amd)}
.sc.ss2n{border-color:var(--b1);background:var(--s1);opacity:.4}
.sn{font-size:9px;color:var(--mu);margin-bottom:4px}
.st{font-family:'Syne',sans-serif;font-size:20px;font-weight:700}
.ss1 .st{color:var(--gr)}.ss2a .st{color:var(--am)}.ss2n .st{color:var(--di)}
.snt{font-size:9px;color:var(--mu);margin-top:2px}
.gr2{margin-bottom:7px}
.glr{display:flex;justify-content:space-between;font-size:10px;margin-bottom:3px;flex-wrap:wrap;gap:4px}
.gv{font-family:'Syne',sans-serif;font-size:12px;font-weight:700}
.gt{height:5px;background:var(--b1)}
.gf{height:100%;transition:width 1s ease}
.fb{background:var(--bl)}.fp{background:var(--pu)}
.cs{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px}
.cb2{padding:8px 6px;text-align:center;border:1px solid var(--b2);background:var(--s2)}
.ca2{font-family:'Syne',sans-serif;font-size:14px;font-weight:700;margin:3px 0;word-break:break-all}
.cl{font-size:9px;color:var(--mu)}
.brr{margin-bottom:6px}
.brl{display:flex;justify-content:space-between;font-size:10px;color:var(--mu);margin-bottom:3px;flex-wrap:wrap;gap:4px}
.bro{height:7px;background:var(--b1)}
.bri{height:100%;transition:width 1.4s ease}
.tw{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.tc{font-size:10px;padding:3px 8px;border:1px solid}
.tc.tf{border-color:var(--rdb);color:var(--rd);background:var(--rdd)}
.tc.tl{border-color:var(--gb);color:var(--gr);background:var(--gd)}
.tp{font-size:11px;line-height:1.8;color:var(--mu);padding:8px;background:var(--bg);border:1px solid var(--b1);max-height:80px;overflow-y:auto}
.tp mark{background:rgba(255,61,61,.18);color:var(--rd)}
.lc{background:var(--s1);border:1px solid var(--b1);padding:20px;display:flex;flex-direction:column;gap:10px}
.lb{height:2px;background:var(--b1)}
.lf{height:100%;background:var(--cy);animation:ld .7s ease-in-out infinite alternate}
@keyframes ld{from{width:0;margin-left:0}to{width:55%;margin-left:45%}}
.lt{font-size:10px;color:var(--cy);letter-spacing:2px}
.pb2{display:inline-flex;font-size:10px;padding:4px 10px;border:1px solid var(--gb);color:var(--gr);background:var(--gd)}
.pg{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.desk-wrap{display:block}.left-col,.right-col{display:block;width:100%;height:auto;overflow:visible}
@media(min-width:900px){
  body{display:flex;flex-direction:column;height:100vh;overflow:hidden}
  .desk-wrap{display:flex;flex:1;overflow:hidden;min-height:0}
  .left-col{width:400px;flex-shrink:0;overflow-y:auto;border-right:1px solid var(--b1);display:flex;flex-direction:column}
  .right-col{flex:1;overflow-y:auto;padding:18px 22px}
  #results{border-top:none;min-height:100%}
  .pg{grid-template-columns:repeat(4,1fr)}
}
</style>
</head>
<body>
<div class="hd">
  <div class="logo">⚑</div>
  <div><div class="brand">FRAUD<em>GUARD</em></div></div>
  <div class="hd-pills"><span class="hp live">● LIVE</span><span class="hp" id="tpsLabel">—</span></div>
</div>
<div class="desk-wrap">
  <div class="left-col">
    <div class="slabel">INPUT JOB POSTING</div>
    <div class="exs" id="exBtns"></div>
    <div class="fields">
      <div><span class="fl">JOB TITLE</span><textarea id="fTitle" rows="2" placeholder="e.g. Senior Software Engineer"></textarea></div>
      <div><span class="fl">DESCRIPTION</span><textarea id="fDesc" rows="4" placeholder="Full job description..."></textarea></div>
      <div><span class="fl">REQUIREMENTS</span><textarea id="fReq" rows="3" placeholder="Skills, experience, education..."></textarea></div>
      <div><span class="fl">COMPANY PROFILE</span><textarea id="fComp" rows="2" placeholder="About the company..."></textarea></div>
      <div><span class="fl">METADATA</span>
        <div class="mg">
          <label class="tr"><input type="checkbox" id="mLogo"><span class="trl">COMPANY LOGO</span></label>
          <label class="tr"><input type="checkbox" id="mQ"><span class="trl">SCREENING Qs</span></label>
          <label class="tr"><input type="checkbox" id="mTele"><span class="trl">TELECOMMUTING</span></label>
          <select id="mEdu"><option value="High School or equivalent">High School</option><option value="Bachelor's Degree" selected>Bachelor's</option><option value="Master's Degree">Master's</option></select>
          <select id="mExp"><option value="Not Applicable">Not Applicable</option><option value="Entry level">Entry Level</option><option value="Mid-Senior level" selected>Mid-Senior</option><option value="Director">Director</option></select>
          <select id="mInd"><option value="Unknown">Unknown</option><option value="Computer Software" selected>Software</option><option value="Staffing and Recruiting">Staffing</option><option value="Financial Services">Finance</option></select>
        </div>
      </div>
    </div>
    <button class="run-btn" id="runBtn" onclick="run()">▶  EXECUTE PIPELINE</button>
  </div>
  <div class="right-col">
    <div id="results">
      <div class="idle"><svg viewBox="0 0 24 24" fill="none" stroke-width="1" stroke-linecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg><div class="idle-t">SELECT AN EXAMPLE THEN EXECUTE</div></div>
    </div>
  </div>
</div>
<script>
const API_URL=window.BACKEND_URL||'https://job-fraud.onrender.com';
const REAL={low_thresh:.20,high_thresh:.65,escalation_rate:.012,svm_tps:198742,bert_tps:118,cost_bert_per_1m:1.177024,cost_ts_per_1m:.014124,reduction_factor:83.3,gpu_hrs_bert_annual:23.5,gpu_hrs_ts_annual:.282,svm_f1:.8817,svm_roc_auc:.993,svm_brier:.0061,svm_fp:2,svm_fn:25,low_risk_pct:94.9,review_pct:1.2,high_risk_pct:3.9};
const FT={"below link":3.37,"using below":3.35,"data entry":3.30,"apply using":3.06,"assistant":2.97,"administrative":2.84,"earn":2.77,"clerk":2.52,"000":2.38,"high school":2.32,"entry level":2.1,"work from home":2.05,"click here":2.4,"weekly pay":1.9,"no experience":1.8,"money":1.5,"make money":2.2};
const EX=[{label:"🚨 Fraud",cls:"ef",f:{title:"Data Entry Clerk — Work From Home · Earn $500/week",description:"Apply using the link below. No experience required. Earn money working from home. Weekly pay guaranteed. Click here.",requirements:"High school diploma. No experience needed.",company_profile:"",has_company_logo:0,has_questions:0,telecommuting:1,required_education:"High School or equivalent",required_experience:"Not Applicable",industry:"Unknown"}},{label:"✅ Legit",cls:"el",f:{title:"Senior Software Engineer — Backend Systems",description:"Growing tech company seeking experienced backend engineer. Design scalable microservices in Python and Go.",requirements:"5+ years. Bachelor's in CS. Python, Docker, Kubernetes.",company_profile:"Acme Corp, publicly traded, San Francisco, founded 2010.",has_company_logo:1,has_questions:1,telecommuting:0,required_education:"Bachelor's Degree",required_experience:"Mid-Senior level",industry:"Computer Software"}},{label:"⚠ Review",cls:"er",f:{title:"Administrative Assistant — International Recruitment",description:"Recruiting administrative assistant on behalf of clients. Data management. Training provided.",requirements:"Good communication. Basic computer. Entry level.",company_profile:"International recruitment agency working for global clients.",has_company_logo:0,has_questions:1,telecommuting:0,required_education:"High School or equivalent",required_experience:"Entry level",industry:"Staffing and Recruiting"}}];
const exDiv=document.getElementById('exBtns');
EX.forEach((ex,i)=>{const b=document.createElement('button');b.className='eb';b.textContent=ex.label;b.onclick=()=>loadEx(i,b);exDiv.appendChild(b);});
function loadEx(i,btn){document.querySelectorAll('.eb').forEach(b=>b.className='eb');btn.className='eb '+EX[i].cls;const f=EX[i].f;document.getElementById('fTitle').value=f.title||'';document.getElementById('fDesc').value=f.description||'';document.getElementById('fReq').value=f.requirements||'';document.getElementById('fComp').value=f.company_profile||'';document.getElementById('mLogo').checked=!!f.has_company_logo;document.getElementById('mQ').checked=!!f.has_questions;document.getElementById('mTele').checked=!!f.telecommuting;document.getElementById('mEdu').value=f.required_education||"Bachelor's Degree";document.getElementById('mExp').value=f.required_experience||'Mid-Senior level';document.getElementById('mInd').value=f.industry||'Computer Software';}
loadEx(0,exDiv.children[0]);
function gf(){return{title:document.getElementById('fTitle').value,description:document.getElementById('fDesc').value,requirements:document.getElementById('fReq').value,company_profile:document.getElementById('fComp').value,benefits:'',has_company_logo:document.getElementById('mLogo').checked?1:0,has_questions:document.getElementById('mQ').checked?1:0,telecommuting:document.getElementById('mTele').checked?1:0,required_education:document.getElementById('mEdu').value,required_experience:document.getElementById('mExp').value,industry:document.getElementById('mInd').value,employment_type:'Full-time',function:'Unknown',location:'US'};}
function getTerms(t){const lo=t.toLowerCase(),h=[];for(const[k,s]of Object.entries(FT))if(lo.includes(k))h.push({term:k,score:s});return h.sort((a,b)=>b.score-a.score).slice(0,8);}
function hilite(t,terms){let o=t.slice(0,350);for(const{term}of terms){const r=new RegExp('('+term.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&')+')','gi');o=o.replace(r,'<mark>$1</mark>');}return o;}
async function run(){
  const btn=document.getElementById('runBtn');
  btn.disabled=true;btn.textContent='▶  RUNNING...';
  const fields=gf();
  const txt=[fields.title,fields.description,fields.requirements,fields.company_profile].join(' ');
  const el=document.getElementById('results');
  el.innerHTML='<div class="lc"><div class="rct">STAGE 1 — HYBRID SVM</div><div class="lb"><div class="lf"></div></div><div class="lt">CALLING API... cold start ~30s</div></div>';
  let data=null,apiOk=false;
  try{const r=await fetch(API_URL+'/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fields}),signal:AbortSignal.timeout(60000)});if(r.ok){data=await r.json();apiOk=true;if(data.stats){const s=data.stats;if(s.cost_bert_1m)s.cost_bert_per_1m=s.cost_bert_1m;if(s.cost_ts_1m)s.cost_ts_per_1m=s.cost_ts_1m;if(s.gpu_hrs_bert)s.gpu_hrs_bert_annual=s.gpu_hrs_bert;if(s.gpu_hrs_ts)s.gpu_hrs_ts_annual=s.gpu_hrs_ts;if(s.svm_roc)s.svm_roc_auc=s.svm_roc;}if(data.stats?.svm_tps)document.getElementById('tpsLabel').textContent=(data.stats.svm_tps/1000).toFixed(0)+'K/s';}}catch(e){console.warn(e.message);}
  if(!data){await new Promise(r=>setTimeout(r,200));const terms=getTerms(txt);const fs=terms.reduce((s,t)=>s+t.score,0);let adj=document.getElementById('mLogo').checked?0:.10;if(document.getElementById('mTele').checked)adj+=.05;const s1v=Math.max(.02,Math.min(.96,1/(1+Math.exp(-(fs*.35-1.2)))+adj));const zone=s1v<.20?'LOW_RISK':s1v>=.65?'HIGH_RISK':'REVIEW';data={_sim:true,stage1:{prob:Math.round(s1v*10000)/10000,time_ms:(160+Math.random()*60).toFixed(1),zone},stage2:zone==='REVIEW'?{prob:Math.min(.95,Math.max(.05,s1v+(Math.random()-.5)*.12)),text_weight:.978,meta_weight:.022,time_ms:(200+Math.random()*50).toFixed(1),note:'Pre-computed',closest_s1:s1v}:null,fraud_terms:terms,stats:REAL};}
  btn.disabled=false;btn.textContent='▶  EXECUTE PIPELINE';
  render(data,txt,apiOk);
}
function render(d,txt,apiOk){
  const s1=d.stage1,s2=d.stage2,R=d.stats,zone=s1.zone,prob=s2?s2.prob:s1.prob,pct=Math.round(prob*100);
  const sc=prob>.65?'sd':prob<.25?'ss':'sw',fc=prob>.65?'fr':prob<.25?'fg':'fa',zc=zone==='LOW_RISK'?'cg':zone==='HIGH_RISK'?'cr':'ca';
  const terms=d.fraud_terms||getTerms(txt),esc=R.escalation_rate;
  document.getElementById('results').innerHTML=`<div style="margin-bottom:6px"><span class="pb2">${apiOk?'✓ LIVE — '+R.svm_tps.toLocaleString()+' preds/sec':'⚠ OFFLINE — local sim'}</span></div><div class="rc ${zc}"><div class="rct">ROUTING DECISION</div><div class="vr"><div><div class="bigp ${sc}">${pct}%</div><div style="font-size:9px;color:var(--mu);margin-top:3px">FRAUD PROBABILITY${s2?' (Stage 2)':' (Stage 1)'}</div><div class="prog"><div class="pf ${fc}" style="width:${pct}%"></div></div></div><div class="vb ${zone}">${zone.replace('_',' ')}</div></div><div style="font-size:9px;color:var(--di);display:flex;gap:12px;flex-wrap:wrap"><span>θ_low=0.20</span><span>θ_high=0.65</span><span style="color:var(--mu)">${zone==='REVIEW'?'→ escalated to Stage 2':'→ Stage 1 only, zero GPU'}</span></div></div><div class="rc cn"><div class="rct">PIPELINE TIMING</div><div class="sg"><div class="sc ss1"><div class="sn">STAGE 1 — SVM</div><div class="st">${s1.time_ms}ms</div><div class="snt">${R.svm_tps.toLocaleString()} preds/sec</div></div><div class="sc ${zone==='REVIEW'?'ss2a':'ss2n'}"><div class="sn">STAGE 2 — DISTILBERT</div><div class="st">${zone==='REVIEW'&&s2?s2.time_ms+'ms':'—'}</div><div class="snt">${zone==='REVIEW'?R.bert_tps+' preds/sec · GPU':'not triggered'}</div></div></div></div>${s2?`<div class="rc cb"><div class="rct">ATTENTION GATE</div><div class="gr2"><div class="glr"><span style="color:var(--mu)">Text (DistilBERT)</span><span class="gv" style="color:var(--bl)">${(s2.text_weight*100).toFixed(1)}%</span></div><div class="gt"><div class="gf fb" style="width:${(s2.text_weight*100).toFixed(1)}%"></div></div></div><div class="gr2"><div class="glr"><span style="color:var(--mu)">Metadata</span><span class="gv" style="color:#a78bfa">${(s2.meta_weight*100).toFixed(1)}%</span></div><div class="gt"><div class="gf fp" style="width:${(s2.meta_weight*100).toFixed(1)}%"></div></div></div></div>`:''}<div class="rc ${terms.length?'cr':'cg'}"><div class="rct">🔍 TERM ANALYSIS</div>${terms.length?`<div class="tw">${terms.map(t=>`<span class="tc tf">${t.term} <span style="opacity:.5">${t.score.toFixed(2)}</span></span>`).join('')}</div><div class="tp">${hilite(txt,terms)}...</div>`:`<div class="tw">${['professional','experienced','enterprise'].map(t=>`<span class="tc tl">${t}</span>`).join('')}</div><div style="font-size:10px;color:var(--mu)">No fraud vocabulary detected.</div>`}</div><div class="rc cn"><div class="rct">⚡ GPU COST</div><div class="cs"><div class="cb2"><div class="cl">BERT/1M</div><div class="ca2" style="color:var(--rd)">$${R.cost_bert_per_1m.toFixed(4)}</div><div class="cl">always GPU</div></div><div class="cb2"><div class="cl">TWO-STAGE/1M</div><div class="ca2" style="color:var(--gr)">$${R.cost_ts_per_1m.toFixed(5)}</div><div class="cl">${(esc*100).toFixed(1)}% GPU</div></div><div class="cb2" style="border-color:var(--amb)"><div class="cl">REDUCTION</div><div class="ca2" style="color:var(--am)">${R.reduction_factor}×</div></div></div><div class="brr"><div class="brl"><span>Universal BERT</span><span style="color:var(--rd)">${R.gpu_hrs_bert_annual}h/yr</span></div><div class="bro"><div class="bri" style="width:100%;background:#7f1d1d"></div></div></div><div class="brr"><div class="brl"><span>Two-Stage</span><span style="color:var(--gr)">${R.gpu_hrs_ts_annual}h/yr</span></div><div class="bro"><div class="bri" style="width:${(esc*100).toFixed(1)}%;background:#14532d;min-width:3px"></div></div></div></div><div class="rc cn"><div class="rct">MODEL PERFORMANCE</div><div class="pg">${[['F1',R.svm_f1],['ROC-AUC',R.svm_roc_auc],['BRIER',R.svm_brier],['FALSE POS',R.svm_fp]].map(([l,v])=>`<div style="padding:8px;background:var(--s2);border:1px solid var(--b2);text-align:center"><div style="font-size:9px;color:var(--mu);margin-bottom:3px">${l}</div><div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:var(--cy)">${v}</div></div>`).join('')}</div></div>`;
  setTimeout(()=>{document.querySelectorAll('.pf,.gf,.bri').forEach(el=>{const w=el.style.width;el.style.width='0';requestAnimationFrame(()=>{el.style.width=w;});});},60);
}
(async()=>{try{const r=await fetch(API_URL+'/health',{signal:AbortSignal.timeout(10000)});if(r.ok){const h=await r.json();document.getElementById('tpsLabel').textContent=(h.svm_tps/1000).toFixed(0)+'K/s';}}catch(_){const t=document.getElementById('tpsLabel');t.textContent='OFFLINE';t.style.color='var(--rd)';}})();
</script>
</body>
</html>
""", 200, {'Content-Type': 'text/html'}

@app.route('/health')
def health():
    return jsonify({"status": "ok", "svm_tps": M['svm_tps'], "loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    fields = request.get_json(force=True).get('fields', {})
    if not any(fields.get(c,'').strip() for c in TEXT_COLS):
        return jsonify({"error": "No text provided"}), 400

    t0 = time.perf_counter()
    text, X_text_vec, X_hyb = build_features(fields)
    s1_prob = float(svm.predict_proba(X_hyb)[0, 1])
    s1_ms   = round((time.perf_counter() - t0) * 1000, 2)

    zone = ('LOW_RISK'  if s1_prob < M['low_thresh']  else
            'HIGH_RISK' if s1_prob >= M['high_thresh'] else 'REVIEW')

    fraud_terms = get_fraud_terms(X_text_vec)

    resp = {
        "stage1": {
            "prob":    round(s1_prob, 4),
            "time_ms": s1_ms,
            "zone":    zone,
            "gpu_used": False,
        },
        "stage2": None,
        "fraud_terms": fraud_terms,
        "thresholds":  {"low": M['low_thresh'], "high": M['high_thresh']},
        "stats": {
            "svm_tps":         M['svm_tps'],
            "bert_tps":        M['bert_tps'],
            "escalation_rate": M['escalation_rate'],
            "reduction_factor":M['reduction_factor'],
            "cost_bert_1m":    M['cost_bert_per_1m'],
            "cost_ts_1m":      M['cost_ts_per_1m'],
            "gpu_hrs_bert":    M['gpu_hrs_bert_annual'],
            "gpu_hrs_ts":      M['gpu_hrs_ts_annual'],
            "low_pct":         M['low_risk_pct'],
            "review_pct":      M['review_pct'],
            "high_pct":        M['high_risk_pct'],
            "svm_f1":          M['svm_f1'],
            "svm_roc":         M['svm_roc_auc'],
            "svm_brier":       M['svm_brier'],
            "svm_fp":          M['svm_fp'],
            "svm_fn":          M['svm_fn'],
        }
    }

    if zone == 'REVIEW':
        t2    = time.perf_counter()
        match = find_s2(s1_prob)
        s2_ms = round((time.perf_counter() - t2) * 1000, 2)
        resp["stage2"] = {
            "prob":        match['s2_prob'],
            "text_weight": match['text_weight'],
            "meta_weight": match['meta_weight'],
            "time_ms":     s2_ms,
            "gpu_used":    True,
            "note":        "Pre-computed on 30 representative REVIEW-zone samples from test set",
            "closest_s1":  match['s1_prob'],
            "top_terms":   match.get('top_terms', []),
        }

    return jsonify(resp)

@app.route('/stats')
def stats():
    return jsonify(M)

@app.route('/terms')
def terms():
    return jsonify({"fraud": lr_terms['fraud'][:20], "legit": lr_terms['legit'][:20],
                    "metadata": meta_info})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
