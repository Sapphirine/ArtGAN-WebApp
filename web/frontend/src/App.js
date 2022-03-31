import './App.css';
import React from 'react'
import { BrowserRouter, Route, Routes, Link } from 'react-router-dom'


import RandomPic from './RandomPic';
import Model from './Model'

function App() {
  return (
    <div className="App">
      <h1> Welcome to the ArtGAN </h1>
      <BrowserRouter>
        <nav>
          <ul>
          <li> <Link to="/dataset">dataset</Link></li>
          <li><Link to="/model">model</Link></li>
          </ul>
        </nav>

        <Routes>
          <Route path="/dataset" element={<RandomPic />}/>
          <Route path="/model" element={<Model />}/>
        </Routes>
      </BrowserRouter>
    </div>
  )
}
export default App;
