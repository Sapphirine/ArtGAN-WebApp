import './App.css';
import React from 'react'
import Navbar from './components/Navbar';
import { BrowserRouter, Route, Routes, Link } from 'react-router-dom'


import RandomPic from './pages/RandomPic';
import Model from './pages/Model';
import Main from './pages/Main';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path='/' element={<Main />}/>
          <Route path='/dataset' element={<RandomPic />}/>
          <Route path='/model' element={<Model />}/>
        </Routes>
      </BrowserRouter>
    </div>
  )
}
export default App;

