const plotLayout = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: {color:"#b9c8da", family:"Inter"}, margin:{l:45,r:15,t:15,b:40},
  xaxis:{gridcolor:"#1a2d43",zerolinecolor:"#1a2d43"},
  yaxis:{gridcolor:"#1a2d43",zerolinecolor:"#1a2d43"}
};
const config = {responsive:true, displayModeBar:false};
async function getJSON(url){const r=await fetch(url); if(!r.ok) throw new Error(await r.text()); return r.json();}
function kpi(label,value,sub=""){return `<div class="kpi"><span>${label}</span><strong>${value}</strong>${sub?`<small style="color:#71859d">${sub}</small>`:""}</div>`;}

async function loadHome(){
  try{
    const m=await getJSON("/api/metrics"), a=await getJSON("/api/analytics"), f=await getJSON("/api/feature-importance");
    const final=m.metrics[m.final_model];
    document.querySelector("#kpis").innerHTML=[
      kpi("Dataset",m.dataset_rows+" rows"), kpi("Features",m.features), kpi("Final model",m.final_model),
      kpi("R²",final.R2.toFixed(2)), kpi("MAE","₹"+final.MAE.toLocaleString("en-IN",{maximumFractionDigits:2})),
      kpi("RMSE","₹"+final.RMSE.toLocaleString("en-IN",{maximumFractionDigits:2})), kpi("Train",m.train_samples), kpi("Test",m.test_samples)
    ].join("");
    Plotly.newPlot("priceChart",[{x:a.price_distribution,type:"histogram",nbinsx:28}],{...plotLayout,xaxis:{...plotLayout.xaxis,title:"Price (₹)"},yaxis:{...plotLayout.yaxis,title:"Phones"}},config);
    Plotly.newPlot("importanceChart",[{x:f.slice(0,8).map(x=>x.Importance).reverse(),y:f.slice(0,8).map(x=>x.Feature).reverse(),type:"bar",orientation:"h"}],{...plotLayout},config);
  }catch(e){console.error(e)}
}

async function loadAnalytics(){
  try{
    const a=await getJSON("/api/analytics");
    Plotly.newPlot("brandAvg",[{x:a.brand_average.map(x=>x.name),y:a.brand_average.map(x=>x.value),type:"bar"}],{...plotLayout,yaxis:{...plotLayout.yaxis,tickprefix:"₹"}},config);
    Plotly.newPlot("brandCount",[{x:a.brand_count.map(x=>x.name),y:a.brand_count.map(x=>x.value),type:"bar"}],plotLayout,config);
    scatter("ramPrice",a.ram_price,"RAM (GB)"); scatter("storagePrice",a.storage_price,"Storage (GB)");
    scatter("ratingPrice",a.rating_price,"Rating score"); scatter("batteryPrice",a.battery_price,"Battery (mAh)");
  }catch(e){console.error(e)}
}
function scatter(id,data,title){Plotly.newPlot(id,[{x:data.map(p=>p.x),y:data.map(p=>p.y),mode:"markers",type:"scatter",marker:{size:7,opacity:.65}}],{...plotLayout,xaxis:{...plotLayout.xaxis,title},yaxis:{...plotLayout.yaxis,title:"Price (₹)"}},config)}

async function loadModel(){
  const m=await getJSON("/api/metrics");
  const final=m.metrics[m.final_model];
  document.querySelector("#modelKpis").innerHTML=[
    kpi("Final model",m.final_model), kpi("R²",final.R2.toFixed(2)),
    kpi("MAE","₹"+final.MAE.toLocaleString("en-IN",{maximumFractionDigits:2})),
    kpi("RMSE","₹"+final.RMSE.toLocaleString("en-IN",{maximumFractionDigits:2}))
  ].join("");
  document.querySelector("#metricsTable").innerHTML=Object.entries(m.metrics)
    .filter(([name])=>name!=="Final Model")
    .map(([name,v])=>`<tr class="${name===m.final_model?'final-row':''}"><td><b>${name}</b></td><td>₹${v.MAE.toLocaleString("en-IN",{maximumFractionDigits:2})}</td><td>₹${v.RMSE.toLocaleString("en-IN",{maximumFractionDigits:2})}</td><td>${v.R2.toFixed(2)}</td><td>${name===m.final_model?'<span class="status-pill">FINAL</span>':'Comparison'}</td></tr>`).join("");
}

async function loadFeatures(){
  const f=await getJSON("/api/feature-importance");
  Plotly.newPlot("fullImportance",[{x:f.map(x=>x.Importance).reverse(),y:f.map(x=>x.Feature).reverse(),type:"bar",orientation:"h"}],{...plotLayout,height:600,xaxis:{...plotLayout.xaxis,title:"Importance"}},config);
}
async function loadDataset(){
  try{
    const a=await getJSON("/api/analytics"),m=await getJSON("/api/metrics");
    document.querySelector("#datasetKpis").innerHTML=[kpi("Rows",m.dataset_rows),kpi("Original columns",27),kpi("Training features",m.features),kpi("Processed features",m.processed_features)].join("");
    const names=m.features ? ["clock_speed_ghz","storage_gb","ram_gb","rating_score","res_width_px","res_height_px","display_inches","battery_mah","charging_watt","rear_camera_count","front_camera_count","smartphone_brand","processor_brand","processor_name","os_name","has_5g","has_nfc","has_ir_blaster","fast_charging","memory_card_supported","memory_card_type","total_camera_count","screen_pixels","ppi","charging_efficiency","premium_features"] : [];
    document.querySelector("#featureList").innerHTML=names.map(x=>`<span>${x}</span>`).join("");
  }catch(e){console.error(e)}
}

function setValue(form,key,value){if(form.elements[key] && value!==undefined) form.elements[key].value=value;}
function fillSample(type){
  const f=document.querySelector("#predictForm"); if(!f)return;
  const s={
    budget:{clock_speed_ghz:1.8,storage_gb:64,ram_gb:4,rating_score:60,res_width_px:1600,res_height_px:720,display_inches:6.5,battery_mah:5000,charging_watt:18,rear_camera_count:2,front_camera_count:1},
    mid:{clock_speed_ghz:2.4,storage_gb:128,ram_gb:8,rating_score:75,res_width_px:2400,res_height_px:1080,display_inches:6.6,battery_mah:5000,charging_watt:33,rear_camera_count:3,front_camera_count:1},
    premium:{clock_speed_ghz:3.9,storage_gb:1024,ram_gb:16,rating_score:90,res_width_px:3200,res_height_px:1440,display_inches:6.9,battery_mah:5500,charging_watt:120,rear_camera_count:4,front_camera_count:1}
  }[type];
  Object.entries(s).forEach(([k,v])=>setValue(f,k,v));
}

const form=document.querySelector("#predictForm");
if(form){
  form.addEventListener("submit",async e=>{
    e.preventDefault();
    const result=document.querySelector("#predictionResult");
    result.classList.remove("hidden"); result.innerHTML="<span>Calculating with Gradient Boosting...</span>";
    const data=Object.fromEntries(new FormData(form).entries());
    ["has_5g","has_nfc","has_ir_blaster","fast_charging","memory_card_supported"].forEach(k=>data[k]=Number(data[k]));
    try{
      const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
      const out=await res.json();
      if(!res.ok) throw new Error(out.error||"Prediction failed");
      result.innerHTML=`<div style="color:#91a4bd;font-size:11px">ESTIMATED MOBILE PRICE</div><div class="price">₹${out.price.toLocaleString("en-IN",{maximumFractionDigits:0})}</div><div class="segment">${out.segment}</div><p style="color:#91a4bd">Generated using the final ${out.model} regression model.</p>`;
    }catch(err){result.innerHTML=`<b style="color:#ff8e8e">Prediction error</b><p>${err.message}</p>`}
  });
}

if(document.querySelector("#kpis")) loadHome();
if(document.querySelector("#brandAvg")) loadAnalytics();
if(document.querySelector("#modelKpis")) loadModel();
if(document.querySelector("#fullImportance")) loadFeatures();
if(document.querySelector("#datasetKpis")) loadDataset();
