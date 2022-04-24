import './App.css';
import React from 'react'
import { BrowserRouter, Route, Routes, Link } from 'react-router-dom'


import RandomPic from './RandomPic';
import Model from './Model'

function App() {
  return (
    <div className="App">
      <h1> Welcome to ArtGAN </h1>
      <p class='intro'> Create incredible art image using only a short description. 
        Our great ArtGAN model will help generate amazing AI art for you. </p>
      
      <BrowserRouter>
        <nav>
          <Link to="/dataset"><button>Dataset</button></Link>
          <br />
          <br />
          <Link to="/model"><button>Model</button></Link>
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
