import React from "react";
import { useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Activity, BookOpen, HeartHandshake, History, Home as HomeIcon, Info, Menu, X } from "lucide-react";
import { Analyze, Home, Journal, Dashboard, HistoryPage, About, Resources } from "./pages";

function Layout({ children }) {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const links = [
    ["/", "Home", HomeIcon], ["/analyze", "Analyze", Activity], ["/journal", "Journal", BookOpen],
    ["/dashboard", "Dashboard", Activity], ["/history", "History", History], ["/about", "About", Info],
    ["/resources", "Resources", HeartHandshake],
  ];
  return <div className="app-shell">
    <aside className={open ? "sidebar open" : "sidebar"}>
      <div className="brand"><span className="brand-mark">◌</span><span>Echo<span>Safe</span></span></div>
      <p className="eyebrow">PRIVATE REFLECTION SPACE</p>
      <nav>{links.map(([to, label, Icon]) => <NavLink key={to} to={to} onClick={() => setOpen(false)} className={({isActive}) => isActive ? "active" : ""}><Icon size={18}/>{label}</NavLink>)}</nav>
      <div className="sidebar-note"><strong>Your words stay yours.</strong><p>Nothing is saved unless you explicitly choose to save it.</p></div>
    </aside>
    <main className="main"><header><button className="mobile-menu" onClick={() => setOpen(!open)}>{open ? <X/> : <Menu/>}</button><div className="crumb">EchoSafe <span>/</span> {location.pathname === "/" ? "Welcome" : location.pathname.slice(1)}</div><Link className="header-link" to="/about">How it works ↗</Link></header>{children}</main>
  </div>;
}

export default function App() { return <Layout><Routes><Route path="/" element={<Home/>}/><Route path="/analyze" element={<Analyze/>}/><Route path="/journal" element={<Journal/>}/><Route path="/dashboard" element={<Dashboard/>}/><Route path="/history" element={<HistoryPage/>}/><Route path="/about" element={<About/>}/><Route path="/resources" element={<Resources/>}/></Routes></Layout>; }
