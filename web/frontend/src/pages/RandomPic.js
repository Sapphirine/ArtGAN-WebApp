import React, { useEffect, useState } from 'react'

function RandomPic() {
    const [showImg, setShowImg] = useState(false)
    const[imgValue, setImgValue] = useState("")

    useEffect(() => {
        getRand()
        }, []);

    const getRand = () => {
        fetch("/dataset")
        .then(response => response.json())
        .then(data => {
            setImgValue(data['imgValue'])
            setShowImg(true)
      })
    }
    function NextRand () {
      return (
          <img src={`data:image/png;base64,${imgValue}`} alt="RANDOM"></img>
      )
    }
  
    return (
        <div>
            <p> This page shows some of the training data for our model. Please click the refresh button and have some insight!</p><br />

            {showImg && 
            <>
                <NextRand /> <br />
            </>
            }
            <br />
            <button class="model" onClick={getRand}>refresh</button>
        </div>
    )
  }
  

export default RandomPic;