// =========================
// PREVENT DEFAULT BROWSER
// =========================

window.addEventListener(
    'dragover',
    (e) => {

        e.preventDefault()
    }
)

window.addEventListener(
    'drop',
    (e) => {

        e.preventDefault()
    }
)

// =========================
// DROP AREA
// =========================

const dropArea =
    document.getElementById('dropArea')

// =========================
// DRAG OVER
// =========================

dropArea.addEventListener(
    'dragover',
    (e) => {

        e.preventDefault()

        dropArea.classList.add(
            'dragover'
        )
    }
)

// =========================
// DRAG LEAVE
// =========================

dropArea.addEventListener(
    'dragleave',
    () => {

        dropArea.classList.remove(
            'dragover'
        )
    }
)

// =========================
// DROP FILE
// =========================

dropArea.addEventListener(
    'drop',
    (e) => {

        e.preventDefault()

        e.stopPropagation()

        dropArea.classList.remove(
            'dragover'
        )

        const file =
            e.dataTransfer.files[0]

        if (file) {

            uploadFile(file)
        }
    }
)

// =========================
// CLICK AREA
// =========================

dropArea.addEventListener(
    'click',
    () => {

        openFilePicker()
    }
)

// =========================
// FILE PICKER
// =========================

function openFilePicker() {

    const input =
        document.createElement('input')

    input.type = 'file'

    input.accept = 'image/*'

    input.onchange = () => {

        const file =
            input.files[0]

        if (file) {

            uploadFile(file)
        }
    }

    input.click()
}

// =========================
// MAIN UPLOAD
// =========================

function uploadFile(file) {

    // =========================
    // ELEMENTS
    // =========================

    const uploadScreen =
        document.getElementById('uploadScreen')

    const resultContainer =
        document.getElementById('resultContainer')

    const preview =
        document.getElementById('preview')

    const alertBox =
        document.getElementById('alertBox')

    const alertText =
        document.getElementById('alertText')

    const percentageBadge =
        document.getElementById('percentageBadge')

    const aiPercent =
        document.getElementById('aiPercent')

    const confidenceText =
        document.getElementById('confidenceText')

    const classificationText =
        document.getElementById('classificationText')

    // =========================
    // SHOW RESULT SCREEN
    // =========================

    uploadScreen.classList.add(
        'hidden'
    )

    resultContainer.classList.remove(
        'hidden'
    )

    // =========================
    // IMAGE PREVIEW
    // =========================

    preview.src =
        URL.createObjectURL(file)

    // =========================
    // RESET DATA
    // =========================

    percentageBadge.innerHTML =
        '0%'

    aiPercent.innerHTML =
        '0%'

    confidenceText.innerHTML =
        'Uploading...'

    classificationText.innerHTML =
        'Scanning...'

    alertText.innerHTML =
        'Uploading Image...'

    // =========================
    // FORM DATA
    // =========================

    const formData =
        new FormData()

    formData.append(
        'image',
        file
    )

    // =========================
    // REQUEST
    // =========================

    const xhr =
        new XMLHttpRequest()

    xhr.open(
        'POST',
        '/predict',
        true
    )

    // =========================
    // UPLOAD PROGRESS
    // =========================

    xhr.upload.onprogress = (event) => {

        if (event.lengthComputable) {

            const percent =
                Math.round(
                    (event.loaded / event.total) *
                    100
                )

            percentageBadge.innerHTML =
                `${percent}%`

            alertText.innerHTML =
                `Uploading ${percent}%`
        }
    }

    // =========================
    // RESPONSE
    // =========================

    xhr.onload = () => {

        const result =
            JSON.parse(xhr.responseText)

        const percentage =
            parseFloat(
                result.confidence
            ).toFixed(2)

        // =========================
        // CONFIDENCE LEVEL
        // =========================

        let confidenceLevel =
            'Low'

        if (percentage >= 85) {

            confidenceLevel =
                'High'
        } else if (percentage >= 60) {

            confidenceLevel =
                'Medium'
        }

        // =========================
        // UPDATE UI
        // =========================

        percentageBadge.innerHTML =
            `${percentage}%`

        aiPercent.innerHTML =
            `${percentage}%`

        confidenceText.innerHTML =
            confidenceLevel

        // =========================
        // AI RESULT
        // =========================

        if (result.label === 'AI Art') {

            // THEME
            alertBox.className =
                'alert-box ai-theme'

            // TEXT
            alertText.innerHTML =
                'This image is likely AI Generated'

            classificationText.innerHTML =
                'AI Generated'

            // COLORS
            percentageBadge.style.color =
                '#b91c1c'

            aiPercent.style.color =
                '#dc2626'

            classificationText.style.color =
                '#dc2626'
        }

        // =========================
        // HUMAN RESULT
        // =========================
        else {

            // THEME
            alertBox.className =
                'alert-box human-theme'

            // TEXT
            alertText.innerHTML =
                'This image is likely Human Illustration'

            classificationText.innerHTML =
                'Human Illustration'

            // COLORS
            percentageBadge.style.color =
                '#166534'

            aiPercent.style.color =
                '#16a34a'

            classificationText.style.color =
                '#16a34a'
        }

        // =========================
        // CONFIDENCE COLOR
        // =========================

        if (result.label === 'AI Art') {

            // =====================
            // AI COLORS
            // =====================

            if (confidenceLevel === 'High') {

                confidenceText.style.color =
                    '#dc2626'
            } else if (confidenceLevel === 'Medium') {

                confidenceText.style.color =
                    '#f97316'
            } else {

                confidenceText.style.color =
                    '#fca5a5'
            }
        } else {

            // =====================
            // HUMAN COLORS
            // =====================

            if (confidenceLevel === 'High') {

                confidenceText.style.color =
                    '#16a34a'
            } else if (confidenceLevel === 'Medium') {

                confidenceText.style.color =
                    '#84cc16'
            } else {

                confidenceText.style.color =
                    '#86efac'
            }
        }
    }

    // =========================
    // ERROR
    // =========================

    xhr.onerror = () => {

        alert(
            'Upload gagal.'
        )
    }

    // =========================
    // SEND
    // =========================

    xhr.send(formData)
}

// =========================
// RESET UI
// =========================

function resetUI() {

    const uploadScreen =
        document.getElementById('uploadScreen')

    const resultContainer =
        document.getElementById('resultContainer')

    uploadScreen.classList.remove(
        'hidden'
    )

    resultContainer.classList.add(
        'hidden'
    )
}