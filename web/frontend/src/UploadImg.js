import React, { useEffect, useState } from 'react'
import Histogram from 'react-chart-histogram'

const class_to_idx = {'abstract': 0,
                    'animal-painting': 1,
                    'cityscape': 2,
                    'figurative': 3,
                    'flower-painting': 4,
                    'genre-painting': 5,
                    'landscape': 6,
                    'marina': 7,
                    'portrait': 8,
                    'religious-painting': 9}

export default function UploadImg() {
    const [file, setFile] = useState("")
    const [result, setResult] = useState([])
    const [showResult, setShowResult] = useState(false)
    const [image, setImage] = useState("")
    const [showImg, setShowImg] = useState(false)
    const [height, setHeight] = useState(1)
    const [width, setWidth] = useState(1)
    const [winner, setWinner] = useState(0)

    useEffect(() => {
        let img = new Image()
        img.src = image
        img.onload = () => {
            if (img.height > img.width) {
                setWidth(200)
                setHeight(Math.ceil(200 * img.height/img.width))
            } else {
                setHeight(200)
                setWidth(Math.ceil(200 * img.width/img.height))
            }
        }
    }, [image])

    function handleImageChange(e) {
        e.preventDefault()
        setFile(e.target.files[0])
        if (file === "") {return}
        setImage(URL.createObjectURL(e.target.files[0]))
        setShowImg(true)
        setShowResult(false)
    }      

    function getKeyByValue(value) {
        return Object.keys(class_to_idx).find(key => class_to_idx[key] === value);
    }
    const upload_file = (e) => {
        e.preventDefault()
        if (file === "") {return}
        var formData = new FormData();
        formData.append("file", file)

        fetch('http://75.101.224.153:8080/upload_img',{
            'method': "POST",
            'Access-Control-Allow-Origin': "*",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            setWinner(getKeyByValue(data['winner']))
            setResult(data['pred'])
            setShowResult(true)
        })
        .catch(error => {
            console.log(error)
        })
    }

    const ResultHist = () => {
        const xlabel = ['abstract',
        'animal',
        'cityscape',
        'figurative',
        'flower',
        'genre',
        'landscape',
        'marina',
        'portrait',
        'religious']
        return (
            <Histogram
                xLabels={xlabel}
                yValues={result}
                width='400'
                height='200'
            />
        )
    }


    return (
        <>
            <h3>Would you like to Classify an image?</h3>
            <form onSubmit={upload_file}>
                <label>Upload an image you want to classify...</label>
                <div>
                    <input type='file' accept="image/*" onChange={handleImageChange}></input>
                    <button type="submit" class='model'>Upload...</button>
                </div>
                <br />
                { showImg && <img src={image} style={{width, height}} alt="Uploaded..."/> }
                { showResult && <ResultHist />}
            </form>
            { showResult &&<p>"{winner}" ? </p> }
        </>
    )
}
