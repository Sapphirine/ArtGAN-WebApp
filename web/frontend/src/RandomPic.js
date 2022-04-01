import React, { useEffect, useState } from 'react'

function RandomPic() {
    const [showImg, setShowImg] = useState(false)
    const[imgValue, setImgValue] = useState("")

    useEffect(() => {
        getRand()
        }, []);

    const getRand = () => {
        fetch("http://100.26.137.252:8080/dataset")
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
            {showImg && 
            <>
                <NextRand /> <br />
            </>
            }
            <button onClick={getRand}>refresh</button>
        </div>
    )
  }
  

export default RandomPic;