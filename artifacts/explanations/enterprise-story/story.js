'use strict';
// Original SVG story. One evidence model drives scores, weights and outcomes.
const $ = id => document.getElementById(id);
const names = {general:'General Support',billing:'Billing',security:'Security',platform:'Platform Operations'};
const beats = [
  {at:0,end:10,tag:'THE USE CASE',title:'An outage. The wrong team.',description:'A support router sends a telemetry outage to General Support.'},
  {at:10,end:20,tag:'FOLLOW THE SCORES',title:'Why does the wrong team win?',description:'The router adds feature contributions. The highest score chooses the queue, subject to a margin rule.'},
  {at:20,end:30,tag:'WHAT A WEIGHT MEANS',title:'A weight sets an influence.',description:'The same coefficient contributes only when its feature is present.'},
  {at:30,end:40,tag:'THE TRADEOFF',title:'A quick fix can disturb other scores.',description:'Target-only fitting changes coefficients that ordinary support also uses.'},
  {at:40,end:51,tag:'THE CORE IDEA',title:'Move in a direction that leaves them still.',description:'Protection restricts which weight changes the compiler can use.'},
  {at:51,end:61,tag:'THE ACTUAL EDIT',title:'Change the outage influence. Check the result.',description:'One Platform coefficient changes. The other weights and biases stay fixed in this admitted edit.'},
  {at:61,end:71,tag:'THE RESULT & THE VALUE',title:'The right team. Protected scores unchanged.',description:'The selected edit also passes the declared final split, within this synthetic trial.'},
  {at:71,end:81,tag:'WHEN TO STOP',title:'Some requests should produce no rollout.',description:'Protection conflicts and missing feature coverage each block this example.'},
  {at:81,end:90,tag:'RESTORE & GENERALIZE',title:'A reversible edit. A generic method.',description:'Specify the change, protect behavior, check the candidate, and retain a bound inverse.'}
];
const cues = [
  [0,5,'An outage ticket lands with the wrong team.'],
  [5,10,'The policy should send it straight to Platform Operations.'],
  [10,15,'The router gives each team a score. The highest score wins.'],
  [15,20,'For this ticket, General Support scores 2. Platform scores only 0.25.'],
  [20,25,'A weight sets how much a feature contributes to a score.'],
  [25,30,'The outage feature is on here, but off for ordinary support.'],
  [30,35,'A target-only edit improves the outage route.'],
  [35,40,'But it also changes an ordinary ticket’s scores. That fails the protection rule.'],
  [40,45,'Think of an edit as a direction in weight space.'],
  [45,51,'Choose a direction with zero effect on the protected inputs.'],
  [51,56,'Here, the framework changes only the outage-to-Platform weight: 0.25 becomes about 3.22.'],
  [56,61,'Now Platform wins this ticket. The model changes through an explicit, checked weight edit.'],
  [61,66,'Final recognized outages reach Platform. The sampled protected scores stay unchanged.'],
  [66,71,'A direct route could avoid a transfer. Handling time and savings have not been measured.'],
  [71,76,'If a target must also stay fixed, the framework abstains.'],
  [76,81,'If a new symptom is not recognized, the final check rejects the candidate.'],
  [81,85.5,'The bound inverse restores the exact parent model. It cannot undo ticket assignments.'],
  [85.5,90,'The method is generic. This demonstration uses fixed features and a supplied linear model.']
];
let data = globalThis.STORY_DATA, evidence = null, time = 0, playing = false, ready = false, loading = false, lastTick = 0;
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const text = (x,y,value,size=24,cls='',extra='') => `<text x="${x}" y="${y}" font-size="${size}" class="${cls}" ${extra}>${String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}</text>`;
const rect = (x,y,w,h,cls='panel',rx=12,extra='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" class="${cls}" ${extra}/>`;
const line = (x1,y1,x2,y2,cls='line',extra='') => `<path d="M${x1} ${y1}L${x2} ${y2}" class="${cls}" ${extra}/>`;
const dot = (x,y,color='var(--cyan)',r=6) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
const f = n => Number(n).toFixed(2).replace(/\.00$/,'');
const clamp = n => Math.max(0,Math.min(1,n));
const ease = n => {n=clamp(n);return n*n*(3-2*n);};
const phase = (local,start,duration) => reduced ? Number(local>=start) : ease((local-start)/duration);
function ticket(x=30,y=55,w=360,label='Telemetry is down.',detail='Recognized outage',cls='cyan') {
  return rect(x,y,w,245)+text(x+25,y+39,'SUPPORT TICKET',17,'muted')+text(x+25,y+91,label,30)+text(x+25,y+132,detail,23,cls)+line(x+25,y+161,x+w-25,y+161)+text(x+25,y+201,'Structured fields → fixed features',18,'muted');
}
function team(x,y,label,cls='cyan',w=330) {
  return rect(x,y,w,88,'panel',12,`style="stroke:var(--${cls})"`)+text(x+w/2,y+53,label,29,cls,'text-anchor="middle"');
}
function bar(x,y,label,value,color='cyan',width=520) {
  const length = Math.max(0,value/4*width);
  return text(x,y,label,22)+rect(x,y+14,width,22,'panel',3)+`<rect x="${x}" y="${y+14}" width="${length}" height="22" rx="3" fill="var(--${color})"/>`+text(x+width+22,y+33,f(value),27,color);
}
function scoresView(values,heading='Queue scores',highlight='general') {
  return text(525,56,heading,19,'muted')+['general','billing','security','platform'].map((q,i)=>bar(525,97+i*66,names[q],values[i],q===highlight?'green':'cyan',475)).join('')+text(525,396,'Raw score units · lead must be at least 0.5',17,'muted');
}
function render(t) {
  if (!data) return;
  time = Math.max(0,Math.min(90,t));
  const i = beats.findIndex(b=>time<b.end);
  const index = i<0?8:i,b=beats[index],local=time-b.at;
  $('eyebrow').textContent=`${String(index+1).padStart(2,'0')} / ${b.tag}`;
  $('title').textContent=b.title; $('description').textContent=b.description;
  $('figure-title').textContent=b.title; $('figure-desc').textContent=b.description;
  const cue=cues.find(c=>time>=c[0]&&time<c[1])||cues[cues.length-1];
  $('caption').textContent=cue[2]; $('time').value=time;
  $('clock').textContent=`${Math.floor(time/60)}:${String(Math.floor(time%60)).padStart(2,'0')} / 1:30`;
  let svg='';
  if(index===0){
    const p=phase(local,1,3);
    svg=ticket(20,60,340)+line(360,185,625,185,'target')+text(422,160,'current route',20,'gold')+team(625,141,'General Support','gold',365)+dot(360+265*p,185,'var(--gold)');
    svg+=line(810,229,810,310,'line','stroke-dasharray="7 7"')+text(847,278,'possible transfer',20,'muted')+team(625,310,'Platform Operations','green',365)+text(20,363,'Desired: reach the owning team directly.',24,'green');
  }
  if(index===1){svg=ticket()+scoresView(data.target.before_scores,'This outage ticket: before the edit','general');}
  if(index===2){
    svg=rect(30,45,380,130)+text(55,85,'Ordinary support',27)+text(55,134,'Outage feature = 0',25,'muted');
    svg+=rect(30,217,380,130)+text(55,257,'Telemetry outage',27)+text(55,306,'Outage feature = 1',25,'cyan');
    const p=phase(local,1,3);
    svg+=line(410,109,586,109,'line')+line(410,282,586,282,'signal')+dot(410+176*p,282)+rect(586,63,250,90)+text(711,119,'0 × 0.25 = 0',29,'muted','text-anchor="middle"')+rect(586,236,250,90)+text(711,292,'1 × 0.25 = 0.25',27,'cyan','text-anchor="middle"');
    svg+=text(886,103,'Platform score',22)+text(886,137,'gets no contribution',19,'muted')+text(886,276,'Platform score',22)+text(886,310,'gets +0.25',25,'cyan')+text(600,388,'feature × weight = contribution',29,'math','text-anchor="middle"');
  }
  if(index===3){
    const p=phase(local,3,3),v=data.ordinaryNaive.after_scores[3]*p;
    svg=ticket(30,65,380,'How do I get help?','Ordinary support','muted');
    svg+=text(505,70,'Platform score for ordinary support',27)+bar(505,126,'Before the target-only edit',data.ordinaryNaive.before_scores[3],'cyan',450)+bar(505,214,'After the target-only edit',v,'red',450);
    svg+=text(505,315,'Still routes to General Support.',23)+text(505,354,'But its protected score moved.',25,'red')+text(30,370,'Outage route improves; protection fails.',23,'gold');
  }
  if(index===4){
    const ox=230,oy=338,sx=92,sy=77,d=data.weightAfter-data.weightBefore;
    svg=line(65,oy,600,oy,'axis')+line(ox,370,ox,38,'axis')+text(420,386,'Δ shared-topic weight',20,'muted')+text(80,23,'Δ outage weight',20,'muted');
    svg+=line(ox,oy,ox,48,'protect')+text(257,65,'protected direction',20,'green')+line(ox,oy-3*sy,ox+3*sx,oy,'target','stroke-dasharray="8 6"')+text(482,215,'target shift ≈ 3',19,'gold');
    const naiveX=data.naiveShared,naiveY=data.naiveOutage;
    svg+=dot(ox+naiveX*sx,oy-naiveY*sy,'var(--red)',8)+text(ox+naiveX*sx+20,oy-naiveY*sy+10,'target-only',20,'red');
    const p=phase(local,2,5);
    svg+=line(ox,oy,ox,oy-d*sy*p,'protect','style="stroke-width:5"')+dot(ox,oy-d*sy*p,'var(--green)',9)+text(ox-15,oy+26,'0',18,'muted');
    svg+=rect(708,54,440,270)+text(735,101,'The protected inputs do not move.',25,'green')+text(735,151,'Their score change stays zero.',24)+text(735,213,'Outage influence can still increase.',24,'cyan')+text(735,270,'That is the permitted edit.',26,'green');
    svg+=text(705,373,'Schematic 2D slice · other constraints omitted',17,'muted');
  }
  if(index===5){
    const p=phase(local,1,4),values=data.target.before_scores.map((v,j)=>v+(data.target.after_scores[j]-v)*p);
    svg=rect(30,50,400,265)+text(55,96,'OUTAGE → PLATFORM WEIGHT',18,'muted')+text(60,168,f(data.weightBefore),51,'cyan')+text(192,168,'→',44,'muted')+text(275,168,f(data.weightBefore+(data.weightAfter-data.weightBefore)*p),51,'green')+text(55,228,'One changed coefficient',26,'green')+text(55,274,'Literal edit: head.weight [3, 6]',19,'muted');
    svg+=scoresView(values,'The same outage ticket: after the edit',p>.6?'platform':'general');
  }
  if(index===6){
    svg=text(30,41,'DECLARED FINAL SPLIT',18,'muted')+team(30,79,'General Support','gold',320)+line(350,123,555,123,'protect')+text(418,98,'outage',18,'cyan')+team(555,79,'Platform Operations','green',355);
    svg+=rect(30,210,540,142)+text(55,258,'Protected scores',26)+text(55,310,'Maximum sampled change: '+f(data.protectedDrift),31,'green');
    svg+=rect(615,210,530,142)+text(640,258,'Operational opportunity',26)+text(640,310,'Avoid the extra transfer',31,'cyan')+text(30,399,'Passed for synthetic trial · no customer rollout or measured savings',20,'muted');
  }
  if(index===7){
    if(local<5){
      svg=rect(55,64,360,231)+text(80,109,'The same ticket',29)+text(80,163,'Target: change its score',23,'gold')+text(80,217,'Anchor: keep its score',23,'green');
      svg+=text(524,145,'Δ score = +3',39,'gold')+text(524,217,'Δ score = 0',39,'green')+team(842,131,'Abstain','red',295)+text(55,359,'Both conditions cannot hold for the same fixed input.',26,'red');
    }else{
      svg=ticket(30,55,400,'Intermittent errors','Not recognized as outage','red')+line(430,177,608,177,'line')+text(501,152,'feature = 0',20,'red');
      svg+=team(608,133,'General Support','gold',350)+text(608,271,'Policy still needs Platform.',26)+text(608,324,'Final check: reject the candidate.',27,'red')+text(30,389,'A weight edit cannot supply missing feature coverage.',25,'muted');
    }
  }
  if(index===8){
    if(local<4.5){
      svg=team(50,93,'Edited model','green',310)+line(360,137,655,137,'signal')+text(408,106,'bound inverse',22,'cyan')+team(655,93,'Exact parent restored','cyan',435);
      svg+=text(50,247,'Outage-to-Platform weight',25)+text(50,312,f(data.weightAfter)+' → '+f(data.weightBefore),43,'cyan')+text(600,250,data.restored?'Artifact identity verified':'See recorded inverse receipt',27,'green')+text(600,301,'External ticket actions are outside scope.',22,'muted');
    }else{
      const labels=['Specify','Edit','Check','Restore'];
      svg=labels.map((l,i)=>team(30+i*293,119,l,i===0?'gold':i===2?'green':'cyan',250)+(i<3?line(280+i*293,163,323+i*293,163,'signal'):'')).join('');
      svg+=text(600,285,'A generic framework for explicit weight changes.',29,'math','text-anchor="middle"')+text(600,337,'This demo: fixed features + a supplied linear head.',24,'muted','text-anchor="middle"')+text(600,387,'Exact protection is scoped; broader runtime guarantees need more evidence.',20,'muted','text-anchor="middle"');
    }
  }
  $('figure').innerHTML=`<title id="figure-title">${b.title}</title><desc id="figure-desc">${b.description}</desc>`+svg;
}
function modelFromBundle(b,conflict,gap,restored){
  const c=label=>b.candidates.find(r=>r.label===label);
  const metrics=c('protected-edit').admission.metrics;
  const naive=c('target-only'),w=c('protected-edit').candidate.artifact.tensors['head.weight'];
  const base=b.parent.tensors['head.weight'];
  return {target:metrics.tickets.find(t=>t.ticket_id==='public-outage-standard'),ordinaryNaive:naive.admission.metrics.tickets.find(t=>t.ticket_id==='public-general'),weightBefore:base[3][6],weightAfter:w[3][6],naiveShared:naive.candidate.artifact.tensors['head.weight'][3][0]-base[3][0],naiveOutage:naive.candidate.artifact.tensors['head.weight'][3][6]-base[3][6],protectedDrift:b.session.result.maximum_protected_score_drift,decision:b.final_admission.decision,conflictDecision:conflict.final_admission.decision,gapDecision:gap.final_admission.decision,restored:restored.exact_parent_restored,parent:b.final_admission.parent_sha256,finalReceipt:b.final_admission.sha256};
}
async function api(path,scenario){
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario})});
  const payload=await response.json();if(!response.ok||!payload.ok)throw Error(payload.error||'Framework run failed');return payload.result;
}
async function prepare(){
  if(ready||loading)return;
  loading=true;$('play').disabled=true;$('play').textContent='Checking the model…';
  try{
    if(location.hostname==='127.0.0.1'&&location.port==='8876'){
      const b=await api('/api/run','policy-change'),conflict=await api('/api/run','conflict'),gap=await api('/api/run','coverage-gap'),restore=await api('/api/rollback','policy-change');
      evidence={policy:b,conflict,gap,restore};data=modelFromBundle(b,conflict,gap,restore);
      if(data.decision!=='admitted_for_synthetic_trial'||!data.restored||data.protectedDrift!==0)throw Error('Unexpected framework outcome; inspect evidence');
      $('run-status').textContent='Framework run verified';
      $('evidence-summary').textContent='This browser session recomputed policy-change, conflict and coverage-gap, then applied the bound inverse. Playback explains these completed results. The full evidence download contains the actual receipts.';
    }
    ready=true;
  }catch(error){$('run-status').textContent='Run failed — playback stopped';$('evidence-summary').textContent=error.message;throw error;}
  finally{loading=false;$('play').disabled=false;$('play').textContent='Play 90-second story';render(time);}
}
function pause(){playing=false;$('play').textContent=time>=90?'Replay story':'Play story';}
function loop(now){if(!playing)return;const elapsed=(now-lastTick)/1000;lastTick=now;render(time+elapsed);if(time>=90){pause();return;}requestAnimationFrame(loop);}
$('play').addEventListener('click',async()=>{if(playing){pause();return;}try{await prepare();if(time>=90)render(0);playing=true;lastTick=performance.now();$('play').textContent='Pause story';requestAnimationFrame(loop);}catch{pause();}});
$('time').addEventListener('input',()=>{pause();render(Number($('time').value));});
// Explicit pointer/keyboard handling also supports browser-control surfaces
// that dispatch input events without applying the native range default action.
function scrub(event){const r=$('time').getBoundingClientRect();pause();render(90*clamp((event.clientX-r.left-8)/(r.width-16)));}
$('time').addEventListener('pointerdown',event=>{event.preventDefault();$('time').focus({preventScroll:true});$('time').setPointerCapture(event.pointerId);scrub(event);});
$('time').addEventListener('pointermove',event=>{if(event.buttons===1)scrub(event);});
$('time').addEventListener('keydown',event=>{const changes={ArrowLeft:-1,ArrowDown:-1,ArrowRight:1,ArrowUp:1,PageDown:-10,PageUp:10};if(event.key in changes||event.key==='Home'||event.key==='End'){event.preventDefault();pause();render(event.key==='Home'?0:event.key==='End'?90:time+changes[event.key]);}});
$('previous').addEventListener('click',()=>{pause();const prev=[...beats].reverse().find(b=>b.at<time-.25);render(prev?prev.at:0);});
$('next').addEventListener('click',()=>{pause();const next=beats.find(b=>b.at>time+.25);render(next?next.at:90);});
$('reset').addEventListener('click',()=>{pause();render(0);});
$('theme').addEventListener('click',()=>{document.body.classList.toggle('light');$('theme').textContent=document.body.classList.contains('light')?'Dark theme':'Light theme';});
$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(evidence||data,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='enterprise-story-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.addEventListener('keydown',e=>{if(['INPUT','BUTTON','SUMMARY'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.code==='ArrowRight'){e.preventDefault();$('next').click();}if(e.code==='ArrowLeft'){e.preventDefault();$('previous').click();}});
render(0);
