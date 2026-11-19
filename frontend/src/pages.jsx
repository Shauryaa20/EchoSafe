import React, { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Check, CircleAlert, LoaderCircle, Lock, Send, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import * as api from "./services/api";

const demo = [
  "I had a normal morning and got some work done.",
  "The messages kept piling up and I could not focus.",
  "I feel tense when I think about everything I still need to do.",
  "I am going to take a break and talk this through with someone I trust.",
];

function Disclaimer() {
  return <div className="disclaimer"><CircleAlert size={18}/><span>EchoSafe identifies text patterns, not a person's mental or clinical state. It is an educational, non-clinical prototype.</span></div>;
}

function Loading() {
  return <div className="loading"><LoaderCircle className="spin" size={20}/>Analyzing language patterns…</div>;
}

function Score({ label, value, tone = "" }) {
  const score = Math.max(0, Math.min(1, Number(value) || 0));
  return <div className="score"><div><span>{label}</span><strong>{Math.round(score * 100)}%</strong></div><div className="meter"><i className={tone} style={{width: `${score * 100}%`}}/></div></div>;
}

function Result({ result, onSave, saved, saveBusy = false, saveError = "" }) {
  const emotions = result.emotions || [];
  const phrases = result.explanations || [];
  return <div className="result">
    <p className="eyebrow accent">LATEST SIGNALS</p>
    <h3>What the text may be expressing</h3>
    <p className="result-caption">Top model scores; this is not an exhaustive clinical assessment.</p>
    <div className="emotion-list">{emotions.slice(0, 5).map((emotion) => <div className="emotion" key={emotion.label}><span>{emotion.label}</span><strong>{Math.round(emotion.score * 100)}%</strong></div>)}</div>
    <Score label="Stress-related language" value={result.stress_score}/>
    <Score label="Model confidence" value={result.uncertainty?.confidence} tone="teal"/>
    {phrases.length > 0 && <div className="phrases"><p className="eyebrow">INFLUENTIAL PHRASES · APPROXIMATE</p>{phrases.slice(0, 4).map((phrase, index) => <span key={`${phrase.text}-${index}`}>{phrase.text}</span>)}</div>}
    <div className="result-note"><Check size={16}/> {result.uncertainty?.is_uncertain ? "The model is uncertain. Treat this as a prompt for reflection, not a conclusion." : "These are model scores about language patterns, not psychological certainty."}</div>
    {onSave && <><button className="button subtle save-button" onClick={onSave} disabled={saved || saveBusy}>{saveBusy ? "Saving…" : saved ? "Saved to history" : "Save analysis (opt in)"}</button>{saved && <div className="save-success">Saved successfully. Open History or Dashboard to view it.</div>}{saveError && <div className="error save-error">{saveError}</div>}</>}
  </div>;
}

export function Home() {
  return <section className="page home-page"><div className="hero"><p className="eyebrow accent">A QUIET PLACE TO NOTICE</p><h1>Understand the patterns <em>behind your words.</em></h1><p className="lede">EchoSafe is a private, non-clinical space for reflecting on the language in your messages — with context, care, and clarity.</p><Link className="button primary" to="/analyze">Start reflecting <ArrowRight size={17}/></Link><p className="micro"><Lock size={13}/> No automatic saving · No diagnosis · No monitoring</p></div><div className="home-grid"><div className="card"><span className="step">01</span><h3>Write freely</h3><p>Share a message or a sequence of messages voluntarily.</p></div><div className="card"><span className="step">02</span><h3>Notice patterns</h3><p>See emotion labels, stress-related language signals, and uncertainty.</p></div><div className="card"><span className="step">03</span><h3>Keep perspective</h3><p>Interpret results as model signals — never as a diagnosis.</p></div></div><Disclaimer/></section>;
}

export function Analyze() {
  const [text, setText] = useState("");
  const [messages, setMessages] = useState([]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const [saveBusy, setSaveBusy] = useState(false);
  const [saveError, setSaveError] = useState("");
  const requestId = useRef(0);

  async function send(value = text) {
    const clean = value.trim();
    if (!clean || busy) return;
    const id = ++requestId.current;
    const context = messages.slice(-5).map((message) => message.text);
    setBusy(true);
    setError("");
    setSaved(false);
    try {
      const data = await api.analyzeText(clean, context);
      if (id !== requestId.current) return;
      setMessages((previous) => [...previous, { text: clean, result: data, context }]);
      setResult(data);
      setText("");
    } catch (requestError) {
      if (id === requestId.current) setError(requestError.message);
    } finally {
      if (id === requestId.current) setBusy(false);
    }
  }

  async function saveCurrent() {
    if (!result || !messages.length || saved) return;
    const latest = messages[messages.length - 1];
    setSaveBusy(true);
    setSaveError("");
    try {
      const savedEntry = await api.saveHistory({
        text: latest.text,
        emotions: result.emotions,
        stress_score: result.stress_score,
        stress_label: result.stress_label,
        uncertainty: result.uncertainty,
        explanations: result.explanations,
        context: latest.context,
        analysis_type: latest.context.length ? "contextual" : "individual",
        model_status: result.model_status,
        mode: result.mode,
      });
      if (!savedEntry?.id) throw new Error("The history database did not return a saved record.");
      setSaved(true);
    } catch (requestError) {
      setSaveError(requestError.message);
    } finally {
      setSaveBusy(false);
    }
  }

  function clearConversation() {
    requestId.current += 1;
    setMessages([]);
    setResult(null);
    setText("");
    setSaved(false);
    setError("");
    setBusy(false);
  }

  return <section className="page"><div className="page-heading"><div><p className="eyebrow accent">LIVE ANALYSIS</p><h2>Make space for the signal.</h2><p>Try a single message, or build a conversation to see how context changes the picture.</p></div><button className="button subtle" onClick={() => setText(demo[0])}>Load demo conversation</button></div><div className="analysis-layout"><div className="chat card"><div className="chat-top"><span className="live-dot"/>Your reflection <button onClick={clearConversation}>Clear</button></div><div className="messages">{messages.length === 0 && <div className="empty-chat"><div className="empty-icon">✦</div><h3>A blank page is a good place to begin.</h3><p>There is no right way to write here. Try describing what is on your mind.</p></div>}{messages.map((message, index) => <div className="message-row" key={`${message.text}-${index}`}><div className="bubble">{message.text}</div><span className="message-time">just now</span></div>)}</div><div className="composer"><textarea value={text} onChange={(event) => setText(event.target.value)} onKeyDown={(event) => {if (event.key === "Enter" && !event.shiftKey) {event.preventDefault(); send();}}} placeholder="Write a message to analyze…" maxLength={5000}/><button className="send" onClick={() => send()} disabled={busy || !text.trim()}>{busy ? <LoaderCircle className="spin"/> : <Send size={18}/>}</button></div><div className="composer-foot"><span>{text.length}/5000</span><span>Enter to send · Shift + Enter for a new line</span></div></div><aside className="insight-panel card">{busy && <Loading/>}{error && <div className="error">{error}</div>}{!busy && !result && !error && <div className="empty-insight"><div className="orb">◌</div><h3>Your insights will appear here.</h3><p>Send a message to see language patterns and model uncertainty.</p></div>}{result && !busy && <Result result={result} onSave={saveCurrent} saved={saved} saveBusy={saveBusy} saveError={saveError}/>}</aside></div><Disclaimer/></section>;
}

export function Journal() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  async function run() {
    if (!text.trim() || busy) return;
    setBusy(true); setError(""); setSaved(false);
    try { setResult(await api.analyzeText(text.trim())); } catch (requestError) { setError(requestError.message); } finally { setBusy(false); }
  }
  async function save() {
    if (!result || saved) return;
    try { await api.saveHistory({text: text.trim(), ...result, analysis_type: "journal", context: []}); setSaved(true); } catch (requestError) { setError(requestError.message); }
  }
  return <section className="page narrow"><div className="page-heading"><div><p className="eyebrow accent">PRIVATE JOURNAL</p><h2>A little more room to reflect.</h2><p>This page does not save your entry unless you choose to later.</p></div></div><div className="journal-layout"><textarea className="journal-input card" value={text} onChange={(event) => {setText(event.target.value); setSaved(false);}} placeholder="Begin wherever feels natural…"/><div className="journal-actions"><button className="button primary" disabled={busy || !text.trim()} onClick={run}>{busy ? <LoaderCircle className="spin"/> : "Analyze entry"}</button><button className="button subtle" onClick={() => {setText(""); setResult(null); setSaved(false);}}>Delete text</button></div>{error && <div className="error">{error}</div>}{busy && <Loading/>}{result && !busy && <div className="card"><Result result={result} onSave={save} saved={saved}/></div>}</div><Disclaimer/></section>;
}

function historyStats(rows) {
  const emotionCounts = {};
  rows.forEach((row) => {
    const top = [...(row.emotions || [])].sort((a, b) => b.score - a.score)[0];
    if (top) emotionCounts[top.label] = (emotionCounts[top.label] || 0) + 1;
  });
  const distribution = Object.entries(emotionCounts).map(([label, count]) => ({label, count})).sort((a, b) => b.count - a.count);
  const timeline = [...rows].reverse().map((row, index) => ({
    name: row.created_at ? new Date(row.created_at).toLocaleDateString() : `#${index + 1}`,
    stress: Math.round((Number(row.stress_score) || 0) * 100),
  }));
  return {distribution, timeline, averageStress: rows.length ? rows.reduce((sum, row) => sum + (Number(row.stress_score) || 0), 0) / rows.length : 0};
}

export function Dashboard() {
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  async function refresh() { setBusy(true); try { const data = await api.getHistory(); setRows(data.entries || []); setError(""); } catch (requestError) { setError(requestError.message); } finally { setBusy(false); } }
  useEffect(() => { refresh(); }, []);
  const stats = useMemo(() => historyStats(rows), [rows]);
  return <section className="page"><div className="page-heading"><div><p className="eyebrow accent">YOUR SESSION</p><h2>A gentle view of change.</h2><p>Signals below are calculated from analyses you explicitly saved.</p></div><button className="button subtle" onClick={refresh} disabled={busy}>{busy ? "Refreshing…" : "Refresh dashboard"}</button></div>{error && <div className="error">{error}</div>}{busy ? <Loading/> : rows.length === 0 ? <div className="card empty-state"><Lock size={20}/><h3>No saved analyses yet.</h3><p>Use “Save analysis (opt in)” after an analysis to build your dashboard.</p></div> : <><div className="stat-grid"><div className="stat card"><span>Total analyses</span><strong>{rows.length}</strong><small>Explicitly saved</small></div><div className="stat card"><span>Conversations</span><strong>{rows.filter((row) => row.analysis_type === "contextual").length}</strong><small>Contextual analyses</small></div><div className="stat card"><span>Average stress signal</span><strong>{Math.round(stats.averageStress * 100)}%</strong><small>Model score, not a clinical measure</small></div></div><div className="dashboard-grid"><div className="chart-card card"><h3>Most frequent top emotion</h3><ResponsiveContainer width="100%" height={240}><BarChart data={stats.distribution}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="label"/><YAxis allowDecimals={false}/><Tooltip/><Bar dataKey="count" fill="#4d9b91" radius={[6, 6, 0, 0]}/></BarChart></ResponsiveContainer></div><div className="chart-card card"><h3>Stress-language trend</h3><ResponsiveContainer width="100%" height={240}><LineChart data={stats.timeline}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name"/><YAxis domain={[0, 100]}/><Tooltip/><Line type="monotone" dataKey="stress" stroke="#d6a257" strokeWidth={3}/></LineChart></ResponsiveContainer></div></div></>}</section>;
}

export function HistoryPage() {
  const [items, setItems] = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  async function refresh() { setBusy(true); try { const data = await api.getHistory(); setItems(data.entries || []); setError(""); } catch (requestError) { setError(requestError.message); } finally { setBusy(false); } }
  useEffect(() => { refresh(); }, []);
  async function remove(id) { try { await api.deleteHistory(id); setItems((current) => current.filter((item) => item.id !== id)); if (expanded === id) setExpanded(null); } catch (requestError) { setError(requestError.message); } }
  async function clear() { if (!items.length || !window.confirm("Delete all saved analyses?")) return; try { await api.clearHistory(); setItems([]); setExpanded(null); } catch (requestError) { setError(requestError.message); } }
  return <section className="page narrow"><div className="page-heading"><div><p className="eyebrow accent">OPT-IN HISTORY</p><h2>Only what you chose to keep.</h2><p>Saved analyses are stored locally unless MongoDB is configured.</p></div><div className="history-actions">{items.length > 0 && <button className="button subtle" onClick={clear}>Clear history</button>}<button className="button subtle" onClick={refresh} disabled={busy}>{busy ? "Refreshing…" : "Refresh history"}</button></div></div>{error && <div className="error">{error}</div>}{busy ? <Loading/> : items.length ? items.map((item) => <div className="history-item card" key={item.id}><div className="history-main" onClick={() => setExpanded(expanded === item.id ? null : item.id)}><p>{item.text}</p><small>{item.created_at ? new Date(item.created_at).toLocaleString() : "Saved entry"} · {item.analysis_type || "individual"}</small>{expanded === item.id && <div className="history-details"><Result result={item}/></div>}</div><button onClick={() => remove(item.id)} aria-label="Delete entry"><Trash2 size={17}/></button></div>) : <div className="card empty-state"><Lock size={20}/><h3>Your history is empty.</h3><p>Analyze a message and explicitly opt in to saving it.</p></div>}</section>;
}

export function About() {
  return <section className="page narrow"><p className="eyebrow accent">THE PROJECT</p><h2>Built to encourage curiosity, not conclusions.</h2><div className="prose card"><h3>What EchoSafe does</h3><p>EchoSafe is an educational deep-learning prototype that analyses voluntarily provided text for emotion categories and stress-related language patterns. It can compare an individual message with its surrounding context and show approximate token-occlusion explanations.</p><h3>What it cannot do</h3><p>It cannot diagnose depression or any illness, predict suicide risk, determine a person's actual emotional condition, or replace qualified support. Scores are model outputs, not facts about you.</p><h3>Architecture</h3><p>The loaded checkpoint uses a CNN and BiGRU message encoder with a contextual LSTM and softmax emotion head. The API reports when the trained checkpoint is unavailable instead of fabricating predictions.</p></div><Disclaimer/></section>;
}

export function Resources() {
  return <section className="page narrow"><p className="eyebrow accent">CARE & SUPPORT</p><h2>You do not have to hold everything alone.</h2><div className="resource-grid"><div className="card"><h3>Talk to someone you trust</h3><p>A friend, family member, mentor, or colleague can be a meaningful first step.</p></div><div className="card"><h3>Consider qualified support</h3><p>A licensed mental health professional can offer guidance matched to your situation.</p></div><div className="card"><h3>Immediate danger</h3><p>If you may be in immediate danger, contact your local emergency services or a trusted person who can be with you.</p></div></div><Disclaimer/></section>;
}
