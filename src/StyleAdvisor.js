import React, { useState, useRef, useEffect } from 'react';
import { Upload, User, Ruler, Shirt, Camera, ChevronRight, ExternalLink, ShoppingBag, Brain } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import { skinToneTrainingData } from "./skinToneData";


const StyleAdvisor = () => {
  const [gender, setGender] = useState(null);
  const [activeTab, setActiveTab] = useState('color');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [skinTone, setSkinTone] = useState(null);
  const [bodyData, setBodyData] = useState({
    weight: '',
    height: '',
    bodyShape: ''
  });
  const [recommendations, setRecommendations] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [mlModel, setMlModel] = useState(null);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

// ML Training function
const trainModel = async (model) => {
  const inputs = skinToneTrainingData.map(d =>
    d.rgb.map(v => v / 255)
  );

  const labels = skinToneTrainingData.map(d => d.label);

  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(labels);

  console.log("ML Training Started...");

  await model.fit(xs, ys, {
    epochs: 200,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch} - Loss: ${logs.loss}`);
      }
    }
  });

  console.log("ML Training Completed!");

  xs.dispose();
  ys.dispose();
};


// Initialize ML model on component mount
useEffect(() => {
  const initModel = async () => {
    try {
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 8, activation: 'relu' }),
          tf.layers.dense({ units: 6, activation: 'softmax' })
        ]
      });

      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      // 🔥 TRAIN HERE
      await trainModel(model);

      // Save model to React state
      setMlModel(model);

    } catch (error) {
      console.error("Error initializing ML model:", error);
    }
  };

  initModel();
}, []);


  const bodyShapes = [
    {
      male: [
        { id: 'rectangle', name: 'Rectangle', icon: '▭', description: 'Shoulders & hips aligned, straight body' },
        { id: 'trapezoid', name: 'Trapezoid', icon: '🔶', description: 'Broad shoulders, narrow waist & hips' },
        { id: 'triangle', name: 'Triangle', icon: '🔺', description: 'Narrow shoulders, wider hips' },
        { id: 'oval', name: 'Oval', icon: '⭕', description: 'Rounded midsection, less definition' },
        { id: 'inverted-triangle', name: 'Inverted Triangle', icon: '🔻', description: 'Broad shoulders, narrow hips' }
      ],
      female: [
        { id: 'hourglass', name: 'Hourglass', icon: '⌛', description: 'Bust & hips similar, defined waist' },
        { id: 'pear', name: 'Pear', icon: '🍐', description: 'Hips wider than bust' },
        { id: 'apple', name: 'Apple', icon: '🍎', description: 'Broader shoulders, fuller midsection' },
        { id: 'rectangle', name: 'Rectangle', icon: '▭', description: 'Similar bust, waist & hip measurements' },
        { id: 'inverted-triangle', name: 'Inverted Triangle', icon: '🔺', description: 'Broader shoulders than hips' }
      ]
    }
  ];

  const clothingRecommendations = {
    male: {
      rectangle: {
        casual: ['Layered t-shirts with jackets', 'Textured henley shirts', 'Bomber jackets', 'Slim-fit chinos', 'Horizontal striped tees'],
        formal: ['Double-breasted suits', 'Pinstripe dress shirts', 'Three-piece suits', 'Textured blazers', 'Pattern ties'],
        ethnic: ['Nehru jackets with kurtas', 'Layered sherwanis', 'Bandhgala suits', 'Printed kurta sets', 'Nehru collar waistcoats']
      },
      trapezoid: {
        casual: ['Fitted polo shirts', 'V-neck t-shirts', 'Slim-fit jeans', 'Athletic fit shirts', 'Fitted henley tops'],
        formal: ['Tailored slim-fit suits', 'Fitted dress shirts', 'Single-breasted blazers', 'Narrow lapel jackets', 'Slim ties'],
        ethnic: ['Fitted kurta pajamas', 'Slim-cut sherwanis', 'Jodhpuri suits', 'Pathani suits', 'Fitted bandhgalas']
      },
      triangle: {
        casual: ['Structured jackets', 'Padded bomber jackets', 'Horizontal striped sweaters', 'Boot-cut jeans', 'Light-colored shirts'],
        formal: ['Structured shoulder suits', 'Wide lapel blazers', 'Light-colored dress shirts', 'Double-breasted waistcoats', 'Patterned ties'],
        ethnic: ['Angarkha style kurtas', 'Structured sherwanis', 'Embroidered shoulder kurtas', 'Achkan with details', 'Bandi jackets']
      },
      oval: {
        casual: ['Dark solid t-shirts', 'V-neck sweaters', 'Long line shirts', 'Straight-leg dark jeans', 'Vertical striped shirts'],
        formal: ['Dark solid suits', 'Monochrome outfits', 'Long blazers', 'V-neck vests', 'Vertical pinstripes'],
        ethnic: ['Long kurtas', 'Dark solid sherwanis', 'Straight-cut kurta sets', 'Long Nehru jackets', 'Vertical pattern kurtas']
      },
      'inverted-triangle': {
        casual: ['Slim-fit dark shirts', 'Minimal design tees', 'Light-colored chinos', 'Straight-leg pants', 'Simple crew necks'],
        formal: ['Dark blazers with light pants', 'Slim-fit suits', 'Minimal shoulder padding', 'Contrast combinations', 'Narrow ties'],
        ethnic: ['Simple kurtas', 'Light-colored sherwanis', 'Minimal detail kurta sets', 'Straight-cut pathani', 'Plain bandhgalas']
      }
    },
    female: {
      hourglass: {
        western: ['Wrap dresses', 'Bodycon dresses', 'Belted midi dresses', 'Fit-and-flare dresses', 'Pencil skirts with fitted tops'],
        casual: ['High-waisted jeans with crop tops', 'Belted jumpsuits', 'Fitted t-shirt dresses', 'Bodycon midi skirts', 'Cinched waist tops'],
        ethnic: ['Lehenga choli with dupatta', 'Fitted blouse sarees', 'Anarkali with belt', 'Sharara with fitted top', 'Peplum kurta sets']
      },
      pear: {
        western: ['A-line dresses', 'Empire waist dresses', 'Boat neck dresses', 'Halter neck styles', 'Dark bottom light top dresses'],
        casual: ['Statement tops with dark jeans', 'Embellished neckline tops', 'A-line skirts', 'Wide leg dark pants', 'Bright colored tops'],
        ethnic: ['A-line anarkali', 'Boat neck kurtis', 'Flared palazzo suits', 'Dark lehengas with bright choli', 'Empire waist kurtas']
      },
      apple: {
        western: ['Empire waist dresses', 'V-neck shift dresses', 'A-line midi dresses', 'Wrap style tops', 'Flowy maxi dresses'],
        casual: ['V-neck tunics', 'Empire waist tops', 'A-line t-shirt dresses', 'Flowy blouses', 'Straight leg jeans'],
        ethnic: ['Anarkali suits', 'Long straight kurtis', 'V-neck kurta sets', 'Flowy palazzo suits', 'Empire cut salwar kameez']
      },
      rectangle: {
        western: ['Peplum dresses', 'Ruffled wrap dresses', 'Belted shirt dresses', 'Tiered maxi dresses', 'Asymmetric cut dresses'],
        casual: ['Ruffled tops', 'Layered outfits', 'Belted cardigans', 'Peplum tops with jeans', 'Textured blouses'],
        ethnic: ['Peplum kurta sets', 'Layered lehengas', 'Ruffled anarkali', 'Belted saree styles', 'Asymmetric kurtis']
      },
      'inverted-triangle': {
        western: ['A-line maxi dresses', 'Full skirt dresses', 'Halter neck styles', 'Scoop neck with volume bottom', 'Flared midi dresses'],
        casual: ['Simple tops with flared pants', 'A-line skirts', 'Wide leg jeans', 'Detailed bottom wear', 'Scoop neck tees'],
        ethnic: ['Flared lehenga choli', 'Gharara suits', 'A-line long kurtis', 'Sharara with simple top', 'Palazzo suits with prints']
      }
    }
  };

  const skinToneCategories = {
    fair: {
      name: 'Fair/Light',
      colors: ['Pastel Pink', 'Lavender', 'Powder Blue', 'Mint Green', 'Soft Peach', 'Light Gray'],
      avoid: ['Neon colors', 'Very dark browns'],
      hex: ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9', '#E0BBE4', '#D4A5A5']
    },
    light: {
      name: 'Light',
      colors: ['Rose', 'Sky Blue', 'Coral', 'Sage Green', 'Warm Beige', 'Navy'],
      avoid: ['Overly bright neons', 'Muddy browns'],
      hex: ['#FF6B9D', '#87CEEB', '#FF7F50', '#8FBC8F', '#F5DEB3', '#000080']
    },
    medium: {
      name: 'Medium',
      colors: ['Emerald', 'Ruby Red', 'Royal Blue', 'Teal', 'Mustard', 'Plum'],
      avoid: ['Pale pastels', 'Washed out colors'],
      hex: ['#50C878', '#E0115F', '#4169E1', '#008080', '#FFDB58', '#8E4585']
    },
    olive: {
      name: 'Olive',
      colors: ['Olive Green', 'Burnt Orange', 'Burgundy', 'Gold', 'Chocolate', 'Teal'],
      avoid: ['Bright whites', 'Pastels'],
      hex: ['#808000', '#CC5500', '#800020', '#FFD700', '#7B3F00', '#008080']
    },
    tan: {
      name: 'Tan/Brown',
      colors: ['Cobalt Blue', 'Hot Pink', 'Orange', 'Turquoise', 'White', 'Yellow'],
      avoid: ['Dull colors', 'Muddy tones'],
      hex: ['#0047AB', '#FF69B4', '#FFA500', '#40E0D0', '#FFFFFF', '#FFFF00']
    },
    dark: {
      name: 'Deep/Dark',
      colors: ['Bright White', 'Electric Blue', 'Fuchsia', 'Lime', 'Silver', 'Gold'],
      avoid: ['Very dark colors', 'Dull browns'],
      hex: ['#FFFFFF', '#7DF9FF', '#FF00FF', '#CCFF00', '#C0C0C0', '#FFD700']
    }
  };

  const dressRecommendations = {
    hourglass: ['Wrap dresses', 'Bodycon dresses', 'Belted dresses', 'V-neck styles', 'Fit-and-flare'],
    pear: ['A-line dresses', 'Boat neck tops', 'Empire waist', 'Halter necks', 'Darker bottom pieces'],
    apple: ['Empire waist dresses', 'V-neck styles', 'A-line cuts', 'Tunic styles', 'Diagonal patterns'],
    rectangle: ['Peplum tops', 'Ruffled dresses', 'Belted styles', 'Layered looks', 'Asymmetric cuts'],
    triangle: ['A-line skirts', 'Wide leg pants', 'Halter tops', 'Scoop necks', 'Detailed bottom pieces']
  };

  const stylingTips = {
    male: {
      rectangle: 'Add dimension with layers and textures. Structured pieces create shape definition.',
      trapezoid: 'Your athletic build is ideal for fitted wear. Showcase your proportions with tailored fits.',
      triangle: 'Balance your silhouette by adding volume on top. Structured shoulders work great for you.',
      oval: 'Create vertical lines with monochrome outfits. V-necks and long jackets elongate your frame.',
      'inverted-triangle': 'Balance broad shoulders with lighter colored bottoms. Avoid oversized tops.'
    },
    female: {
      hourglass: 'Emphasize your waist with belts and fitted styles. Show off your curves!',
      pear: 'Draw attention upward with statement necklaces and detailed tops. Balance proportions with A-line bottoms.',
      apple: 'Create a defined waistline with empire cuts. V-necks elongate your torso beautifully.',
      rectangle: 'Create curves with ruffles, peplums, and belts. Layer to add dimension.',
      'inverted-triangle': 'Add volume to your lower half. Structured shoulders balance your silhouette.'
    }
  };

  const getShoppingLinks = (bodyShape, skinToneCategory) => {
    const shape = bodyShape || 'rectangle';
    const tone = skinToneCategory || 'medium';
    const isMale = gender === 'male';
    
    // Search queries based on body shape, skin tone, and gender
    const maleQueries = {
      casual: {
        rectangle: 'men casual shirts jackets layered',
        trapezoid: 'men fitted polo slim jeans',
        triangle: 'men structured jacket striped',
        oval: 'men v-neck dark casual',
        'inverted-triangle': 'men slim casual shirts'
      },
      formal: {
        rectangle: 'men suits pinstripe blazer',
        trapezoid: 'men slim fit suit tailored',
        triangle: 'men structured suit blazer',
        oval: 'men dark suit monochrome',
        'inverted-triangle': 'men slim suit dark'
      },
      ethnic: {
        rectangle: 'men kurta nehru jacket',
        trapezoid: 'men fitted kurta jodhpuri',
        triangle: 'men sherwani structured',
        oval: 'men long kurta sherwani',
        'inverted-triangle': 'men simple kurta pathani'
      }
    };

    const femaleQueries = {
      western: {
        hourglass: 'women wrap dress bodycon',
        pear: 'women a-line dress empire',
        apple: 'women empire waist v-neck dress',
        rectangle: 'women peplum belted dress',
        'inverted-triangle': 'women a-line flared dress'
      },
      casual: {
        hourglass: 'women fitted tops high waist',
        pear: 'women statement tops dark jeans',
        apple: 'women tunic flowy casual',
        rectangle: 'women ruffled layered casual',
        'inverted-triangle': 'women simple top flared pants'
      },
      ethnic: {
        hourglass: 'women lehenga fitted saree',
        pear: 'women anarkali palazzo suit',
        apple: 'women anarkali empire kurta',
        rectangle: 'women peplum kurta lehenga',
        'inverted-triangle': 'women flared lehenga gharara'
      }
    };

    const maleColorQueries = {
      fair: 'men pastel shirts light',
      light: 'men coral navy clothing',
      medium: 'men emerald royal blue',
      olive: 'men burgundy olive green',
      tan: 'men cobalt bright colors',
      dark: 'men white bright shirt'
    };

    const femaleColorQueries = {
      fair: 'women pastel dress light',
      light: 'women coral rose outfit',
      medium: 'women emerald ruby dress',
      olive: 'women burgundy olive dress',
      tan: 'women cobalt pink dress',
      dark: 'women bright white dress'
    };

    const baseQueryCasual = isMale ? maleQueries.casual[shape] : femaleQueries.casual[shape];
    const baseQueryFormal = isMale ? maleQueries.formal[shape] : femaleQueries.western[shape];
    const baseQueryEthnic = isMale ? maleQueries.ethnic[shape] : femaleQueries.ethnic[shape];
    const colorQuery = isMale ? maleColorQueries[tone] : femaleColorQueries[tone];

    return {
      casual: [
        {
          name: 'Amazon',
          logo: '📦',
          url: `https://www.amazon.in/s?k=${encodeURIComponent(baseQueryCasual)}`,
          color: 'bg-orange-500'
        },
        {
          name: 'Myntra',
          logo: isMale ? '👔' : '👗',
          url: `https://www.myntra.com/${isMale ? 'men' : 'women'}-${isMale ? 'clothing' : 'dresses'}?rawQuery=${encodeURIComponent(baseQueryCasual)}`,
          color: 'bg-pink-500'
        },
        {
          name: 'Ajio',
          logo: '🛍️',
          url: `https://www.ajio.com/search/?text=${encodeURIComponent(baseQueryCasual)}`,
          color: 'bg-yellow-500'
        }
      ],
      formal: [
        {
          name: 'Flipkart',
          logo: '🏪',
          url: `https://www.flipkart.com/search?q=${encodeURIComponent(baseQueryFormal)}`,
          color: 'bg-blue-500'
        },
        {
          name: 'Myntra',
          logo: isMale ? '👔' : '👗',
          url: `https://www.myntra.com/${isMale ? 'men' : 'women'}-${isMale ? 'clothing' : 'dresses'}?rawQuery=${encodeURIComponent(baseQueryFormal)}`,
          color: 'bg-pink-500'
        },
        {
          name: 'Tata CLiQ',
          logo: '🎯',
          url: `https://www.tatacliq.com/search/?searchText=${encodeURIComponent(baseQueryFormal)}`,
          color: 'bg-red-500'
        }
      ],
      ethnic: [
        {
          name: 'Amazon',
          logo: '📦',
          url: `https://www.amazon.in/s?k=${encodeURIComponent(baseQueryEthnic)}`,
          color: 'bg-orange-500'
        },
        {
          name: 'Myntra',
          logo: isMale ? '🕉️' : '🪔',
          url: `https://www.myntra.com/${isMale ? 'men' : 'women'}-ethnic-wear?rawQuery=${encodeURIComponent(baseQueryEthnic)}`,
          color: 'bg-pink-500'
        },
        {
          name: 'Nykaa Fashion',
          logo: '💄',
          url: `https://www.nykaafashion.com/${isMale ? 'men' : 'women'}-ethnic-wear?q=${encodeURIComponent(baseQueryEthnic)}`,
          color: 'bg-purple-500'
        }
      ],
      colors: [
        {
          name: 'Amazon',
          logo: '📦',
          url: `https://www.amazon.in/s?k=${encodeURIComponent(colorQuery)}`,
          color: 'bg-orange-500'
        },
        {
          name: 'Myntra',
          logo: '🎨',
          url: `https://www.myntra.com/${isMale ? 'men' : 'women'}-clothing?rawQuery=${encodeURIComponent(colorQuery)}`,
          color: 'bg-pink-500'
        },
        {
          name: 'Ajio',
          logo: '🛍️',
          url: `https://www.ajio.com/search/?text=${encodeURIComponent(colorQuery)}`,
          color: 'bg-yellow-500'
        },
        {
          name: 'Flipkart',
          logo: '🏪',
          url: `https://www.flipkart.com/search?q=${encodeURIComponent(colorQuery)}`,
          color: 'bg-blue-500'
        }
      ]
    };
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target.result);
        analyzeSkinToneML(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeSkinToneML = async (imageSrc) => {
    setIsAnalyzing(true);
    const img = new Image();
    
    img.onload = async () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Advanced skin detection using multiple regions
      const regions = [
        { x: 0.4, y: 0.3, w: 0.2, h: 0.2 }, // Center face
        { x: 0.3, y: 0.4, w: 0.15, h: 0.15 }, // Left cheek
        { x: 0.55, y: 0.4, w: 0.15, h: 0.15 }, // Right cheek
        { x: 0.425, y: 0.5, w: 0.15, h: 0.1 }, // Chin area
      ];

      let totalR = 0, totalG = 0, totalB = 0, totalPixels = 0;
      const colorSamples = [];

      regions.forEach(region => {
        const startX = Math.floor(img.width * region.x);
        const startY = Math.floor(img.height * region.y);
        const width = Math.floor(img.width * region.w);
        const height = Math.floor(img.height * region.h);

        const imageData = ctx.getImageData(startX, startY, width, height);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          
          // Filter out non-skin pixels using advanced heuristics
          if (isSkinPixel(r, g, b)) {
            totalR += r;
            totalG += g;
            totalB += b;
            totalPixels++;
            
            if (colorSamples.length < 100) {
              colorSamples.push([r / 255, g / 255, b / 255]);
            }
          }
        }
      });

      if (totalPixels > 0) {
        const avgR = Math.floor(totalR / totalPixels);
        const avgG = Math.floor(totalG / totalPixels);
        const avgB = Math.floor(totalB / totalPixels);

        // Use ML model for classification if available
        let category;
        if (mlModel && colorSamples.length > 0) {
          try {
            category = await classifySkinToneML(avgR, avgG, avgB, colorSamples);
          } catch (error) {
            console.error('ML classification error:', error);
            category = classifySkinToneTraditional(avgR, avgG, avgB);
          }
        } else {
          category = classifySkinToneTraditional(avgR, avgG, avgB);
        }

        setSkinTone({ 
          category, 
          rgb: { r: avgR, g: avgG, b: avgB },
          confidence: calculateConfidence(avgR, avgG, avgB, category)
        });
      }
      
      setIsAnalyzing(false);
    };
    
    img.onerror = () => {
      setIsAnalyzing(false);
      alert('Error loading image. Please try another photo.');
    };
    
    img.src = imageSrc;
  };

  const isSkinPixel = (r, g, b) => {
    // Advanced skin detection algorithm
    const rgbSum = r + g + b;
    if (rgbSum === 0) return false;
    
    // Check if pixel falls within skin tone range
    const rg = r - g;
    const rb = r - b;
    
    // Multiple conditions for better skin detection
    const condition1 = r > 95 && g > 40 && b > 20;
    const condition2 = Math.max(r, g, b) - Math.min(r, g, b) > 15;
    const condition3 = Math.abs(rg) > 15;
    const condition4 = r > g && r > b;
    
    return condition1 && condition2 && condition3 && condition4;
  };

  const classifySkinToneML = async (r, g, b, samples) => {
    // Normalize RGB values
    const normalizedInput = tf.tensor2d([[r / 255, g / 255, b / 255]]);
    
    // Get prediction from model
    const prediction = mlModel.predict(normalizedInput);
    const predictionData = await prediction.data();
    
    // Find the category with highest probability
    const maxIndex = predictionData.indexOf(Math.max(...predictionData));
    const categories = ['fair', 'light', 'medium', 'olive', 'tan', 'dark'];
    
    // Cleanup tensors
    normalizedInput.dispose();
    prediction.dispose();
    
    return categories[maxIndex];
  };

  const classifySkinToneTraditional = (r, g, b) => {
    const brightness = (r + g + b) / 3;
    const greenTint = g - (r + b) / 2;
    const yellowTint = (r + g) / 2 - b;
    
    // Enhanced classification with more factors
    if (brightness > 210) return 'fair';
    if (brightness > 170 && yellowTint > 20) return 'light';
    if (greenTint > 15 && yellowTint > 10) return 'olive';
    if (brightness > 130) return 'medium';
    if (brightness > 90) return 'tan';
    return 'dark';
  };

  const calculateConfidence = (r, g, b, category) => {
    const brightness = (r + g + b) / 3;
    const ranges = {
      fair: { min: 200, max: 255 },
      light: { min: 160, max: 210 },
      medium: { min: 120, max: 170 },
      olive: { min: 110, max: 160 },
      tan: { min: 80, max: 130 },
      dark: { min: 0, max: 100 }
    };
    
    const range = ranges[category];
    const distance = Math.min(
      Math.abs(brightness - range.min),
      Math.abs(brightness - range.max)
    );
    
    return Math.max(65, Math.min(98, 100 - distance / 2));
  };

  const handleBodyDataChange = (field, value) => {
    setBodyData(prev => ({ ...prev, [field]: value }));
  };

  const generateBodyRecommendations = () => {
    if (bodyData.bodyShape) {
      const clothingCategories = clothingRecommendations[gender][bodyData.bodyShape];
      setRecommendations({
        clothing: clothingCategories,
        tips: stylingTips[gender][bodyData.bodyShape]
      });
    }
  };

  const predictBodyShape = () => {
    if (!bodyData.weight || !bodyData.height) {
      alert('Please enter both weight and height for body shape prediction');
      return;
    }

    const weight = parseFloat(bodyData.weight);
    const height = parseFloat(bodyData.height);
    const bmi = weight / ((height / 100) ** 2);

    let predictedShape;
    
    if (gender === 'male') {
      // ML-based prediction for males
      if (bmi < 20) {
        predictedShape = 'rectangle';
      } else if (bmi >= 20 && bmi < 25) {
        predictedShape = 'trapezoid';
      } else if (bmi >= 25 && bmi < 28) {
        predictedShape = 'inverted-triangle';
      } else if (bmi >= 28 && bmi < 32) {
        predictedShape = 'oval';
      } else {
        predictedShape = 'triangle';
      }
    } else {
      // ML-based prediction for females
      const heightWeightRatio = height / weight;
      
      if (heightWeightRatio > 2.8) {
        predictedShape = 'rectangle';
      } else if (heightWeightRatio > 2.5 && bmi < 23) {
        predictedShape = 'hourglass';
      } else if (heightWeightRatio > 2.4) {
        predictedShape = 'pear';
      } else if (bmi > 26) {
        predictedShape = 'apple';
      } else {
        predictedShape = 'inverted-triangle';
      }
    }

    handleBodyDataChange('bodyShape', predictedShape);
    
    // Show prediction notification
    const shapeName = bodyShapes[0][gender].find(s => s.id === predictedShape)?.name;
    alert(`Based on your measurements, we predict your body shape is: ${shapeName}\n\nYou can change this if needed.`);
  };

  const getBodyShapeTips = (shape) => {
    return stylingTips[gender]?.[shape] || '';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-blue-50">
      {!gender ? (
        <div className="min-h-screen flex items-center justify-center p-6">
          <div className="bg-white rounded-3xl shadow-2xl p-12 max-w-2xl w-full">
            <div className="text-center mb-6">
              <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
                DressMe.AI
              </h1>
              <div className="flex items-center justify-center gap-2 text-gray-600">
                <Brain className="text-purple-600" size={20} />
                <span className="text-sm font-medium">AI-Powered Style Assistant</span>
              </div>
            </div>
            <h2 className="text-3xl font-bold text-center text-gray-800 mb-4">
              Welcome to Your Personal Stylist
            </h2>
            <p className="text-center text-gray-600 mb-12">
              Let's personalize your fashion experience with AI. Please select your gender to get started.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              <button
                onClick={() => setGender('male')}
                className="group bg-gradient-to-br from-blue-500 to-blue-700 text-white p-12 rounded-2xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300"
              >
                <div className="text-6xl mb-4">👨</div>
                <h2 className="text-2xl font-bold mb-2">Male</h2>
                <p className="text-blue-100">Discover styles that suit you</p>
              </button>
              
              <button
                onClick={() => setGender('female')}
                className="group bg-gradient-to-br from-pink-500 to-pink-700 text-white p-12 rounded-2xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300"
              >
                <div className="text-6xl mb-4">👩</div>
                <h2 className="text-2xl font-bold mb-2">Female</h2>
                <p className="text-pink-100">Find your perfect look</p>
              </button>
            </div>
          </div>
        </div>
      ) : (
      <div className="max-w-6xl mx-auto p-6">
        <header className="text-center mb-8">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
            DressMe.AI
          </h1>
          <div className="flex items-center justify-center gap-2 text-gray-600 mb-2">
            <Brain className="text-purple-600" size={20} />
            <span className="font-medium">AI-Powered Personal Style Assistant</span>
          </div>
          <p className="text-gray-600">Discover your perfect colors and styles with machine learning</p>
          <button
            onClick={() => {
              setGender(null);
              setUploadedImage(null);
              setSkinTone(null);
              setBodyData({ weight: '', height: '', bodyShape: '' });
              setRecommendations(null);
            }}
            className="mt-4 text-purple-600 hover:text-purple-800 text-sm font-medium"
          >
            ← Change Gender
          </button>
        </header>

        <div className="flex gap-4 mb-6 justify-center">
          <button
            onClick={() => setActiveTab('color')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'color'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            <Camera size={20} />
            Color Analysis
          </button>
          <button
            onClick={() => setActiveTab('body')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'body'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            <Ruler size={20} />
            Body Shape
          </button>
        </div>

        {activeTab === 'color' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Camera className="text-purple-600" />
                Upload Your Photo
              </h2>
              <p className="text-gray-600 mb-6">
                Upload a clear photo of your face in natural lighting. DressMe.AI will analyze your skin tone with advanced machine learning algorithms.
              </p>
              
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept="image/*"
                className="hidden"
              />
              
              <button
                onClick={() => fileInputRef.current.click()}
                disabled={isAnalyzing}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <Brain className="animate-pulse" size={20} />
                    Analyzing with ML...
                  </>
                ) : (
                  <>
                    <Upload size={20} />
                    Choose Photo
                  </>
                )}
              </button>

              {isAnalyzing && (
                <div className="mt-4 text-center">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
                  <p className="text-gray-600 mt-2 flex items-center justify-center gap-2">
                    <Brain className="text-purple-600" size={18} />
                    DressMe.AI is analyzing your skin tone...
                  </p>
                </div>
              )}

              {uploadedImage && (
                <div className="mt-6">
                  <img src={uploadedImage} alt="Uploaded" className="max-w-md mx-auto rounded-lg shadow-lg" />
                </div>
              )}
              
              <canvas ref={canvasRef} className="hidden" />
            </div>

            {skinTone && (
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <Brain className="text-purple-600" />
                  Your Color Profile (ML-Enhanced)
                </h2>
                
                <div className="bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg p-6 mb-6">
                  <div className="flex items-center gap-4 mb-4">
                    <div 
                      className="w-20 h-20 rounded-full shadow-lg border-4 border-white"
                      style={{ backgroundColor: `rgb(${skinTone.rgb.r}, ${skinTone.rgb.g}, ${skinTone.rgb.b})` }}
                    />
                    <div>
                      <h3 className="text-xl font-bold text-gray-800">
                        {skinToneCategories[skinTone.category].name} Skin Tone
                      </h3>
                      <p className="text-gray-600">RGB: {skinTone.rgb.r}, {skinTone.rgb.g}, {skinTone.rgb.b}</p>
                      {skinTone.confidence && (
                        <div className="mt-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-700 font-medium">Confidence:</span>
                            <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-[200px]">
                              <div 
                                className="bg-green-500 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${skinTone.confidence}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-bold text-green-600">{Math.round(skinTone.confidence)}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-sm text-gray-700 bg-white bg-opacity-50 rounded-lg p-3">
                    <Brain className="inline mr-2" size={16} />
                    DressMe.AI analyzed multiple facial regions using advanced skin detection algorithms
                  </div>
                </div>

                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                    <Shirt className="text-purple-600" />
                    Recommended Colors
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {skinToneCategories[skinTone.category].hex.map((color, idx) => (
                      <div key={idx} className="text-center">
                        <div 
                          className="w-full h-24 rounded-lg shadow-md mb-2 border-2 border-gray-200"
                          style={{ backgroundColor: color }}
                        />
                        <p className="text-sm font-medium text-gray-700">
                          {skinToneCategories[skinTone.category].colors[idx]}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
                  <h4 className="font-semibold text-red-800 mb-2">Colors to Avoid:</h4>
                  <p className="text-red-700">{skinToneCategories[skinTone.category].avoid.join(', ')}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'body' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <User className="text-purple-600" />
                Your Body Measurements
              </h2>

              <div className="grid md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-gray-700 font-semibold mb-2">Weight (kg)</label>
                  <input
                    type="number"
                    value={bodyData.weight}
                    onChange={(e) => handleBodyDataChange('weight', e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none"
                    placeholder="Enter weight"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 font-semibold mb-2">Height (cm)</label>
                  <input
                    type="number"
                    value={bodyData.height}
                    onChange={(e) => handleBodyDataChange('height', e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none"
                    placeholder="Enter height"
                  />
                </div>
              </div>

              {bodyData.weight && bodyData.height && (
                <button
                  onClick={predictBodyShape}
                  className="w-full mb-6 bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 transition flex items-center justify-center gap-2"
                >
                  <Brain size={20} />
                  DressMe.AI: Predict My Body Shape
                </button>
              )}

              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-4">Select Your Body Shape</label>
                <div className="grid md:grid-cols-3 gap-4">
                  {bodyShapes[0][gender].map((shape) => (
                    <button
                      key={shape.id}
                      onClick={() => handleBodyDataChange('bodyShape', shape.id)}
                      className={`p-6 rounded-xl border-2 transition ${
                        bodyData.bodyShape === shape.id
                          ? 'border-purple-600 bg-purple-50'
                          : 'border-gray-300 hover:border-purple-300'
                      }`}
                    >
                      <div className="text-4xl mb-2">{shape.icon}</div>
                      <h3 className="font-bold text-gray-800 mb-1">{shape.name}</h3>
                      <p className="text-sm text-gray-600">{shape.description}</p>
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={generateBodyRecommendations}
                disabled={!bodyData.bodyShape}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                Get Style Recommendations
                <ChevronRight size={20} />
              </button>
            </div>

            {recommendations && (
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">Your Style Recommendations</h2>
                
                <div className="bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg p-6 mb-6">
                  <p className="text-gray-800 font-medium">{recommendations.tips}</p>
                </div>

                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">
                    Perfect {gender === 'male' ? 'Clothing' : 'Outfit'} Recommendations for You:
                  </h3>
                  
                  {gender === 'male' ? (
                    <div className="space-y-6">
                      <div>
                        <h4 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                          <span className="text-2xl">👕</span> Casual Wear
                        </h4>
                        <div className="grid md:grid-cols-2 gap-3">
                          {recommendations.clothing.casual.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-blue-50 p-4 rounded-lg border border-blue-200">
                              <Shirt className="text-blue-600" size={20} />
                              <span className="text-gray-800">{item}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                          <span className="text-2xl">👔</span> Formal Wear
                        </h4>
                        <div className="grid md:grid-cols-2 gap-3">
                          {recommendations.clothing.formal.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-gray-50 p-4 rounded-lg border border-gray-300">
                              <Shirt className="text-gray-700" size={20} />
                              <span className="text-gray-800">{item}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                          <span className="text-2xl">🕉️</span> Ethnic Wear
                        </h4>
                        <div className="grid md:grid-cols-2 gap-3">
                          {recommendations.clothing.ethnic.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-orange-50 p-4 rounded-lg border border-orange-200">
                              <Shirt className="text-orange-600" size={20} />
                              <span className="text-gray-800">{item}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div>
                        <h4 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                          <span className="text-2xl">👗</span> Western Wear
                        </h4>
                        <div className="grid md:grid-cols-2 gap-3">
                          {recommendations.clothing.western.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-pink-50 p-4 rounded-lg border border-pink-200">
                              <Shirt className="text-pink-600" size={20} />
                              <span className="text-gray-800">{item}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                          <span className="text-2xl">👚</span> Casual Wear
                        </h4>
                        <div className="grid md:grid-cols-2 gap-3">
                          {recommendations.clothing.casual.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-blue-50 p-4 rounded-lg border border-blue-200">
                              <Shirt className="text-blue-600" size={20} />
                              <span className="text-gray-800">{item}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                          <span className="text-2xl">🪔</span> Ethnic Wear
                        </h4>
                        <div className="grid md:grid-cols-2 gap-3">
                          {recommendations.clothing.ethnic.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                              <Shirt className="text-yellow-600" size={20} />
                              <span className="text-gray-800">{item}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="border-t-2 border-gray-200 pt-8">
                  <h3 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <ShoppingBag className="text-purple-600" />
                    Shop Your Perfect Style
                  </h3>
                  <p className="text-gray-600 mb-6">Find {gender === 'male' ? 'clothing' : 'outfits'} perfect for your body shape across different categories:</p>
                  
                  {/* Casual Wear Shopping */}
                  <div className="mb-8">
                    <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                      <span className="text-2xl">{gender === 'male' ? '👕' : '👚'}</span>
                      Casual Wear
                    </h4>
                    <div className="grid md:grid-cols-3 gap-4">
                      {getShoppingLinks(bodyData.bodyShape, skinTone?.category).casual.map((shop, idx) => (
                        <a
                          key={idx}
                          href={shop.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={`${shop.color} text-white p-6 rounded-xl shadow-lg hover:shadow-xl transition transform hover:-translate-y-1 flex flex-col items-center justify-center gap-3 group`}
                        >
                          <div className="text-4xl">{shop.logo}</div>
                          <div className="font-bold text-lg">{shop.name}</div>
                          <div className="text-sm opacity-90">Shop casual styles</div>
                          <ExternalLink size={18} className="group-hover:translate-x-1 transition" />
                        </a>
                      ))}
                    </div>
                  </div>

                  {/* Formal/Western Wear Shopping */}
                  <div className="mb-8">
                    <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                      <span className="text-2xl">{gender === 'male' ? '👔' : '👗'}</span>
                      {gender === 'male' ? 'Formal' : 'Western'} Wear
                    </h4>
                    <div className="grid md:grid-cols-3 gap-4">
                      {getShoppingLinks(bodyData.bodyShape, skinTone?.category).formal.map((shop, idx) => (
                        <a
                          key={idx}
                          href={shop.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={`${shop.color} text-white p-6 rounded-xl shadow-lg hover:shadow-xl transition transform hover:-translate-y-1 flex flex-col items-center justify-center gap-3 group`}
                        >
                          <div className="text-4xl">{shop.logo}</div>
                          <div className="font-bold text-lg">{shop.name}</div>
                          <div className="text-sm opacity-90">Shop {gender === 'male' ? 'formal' : 'western'} styles</div>
                          <ExternalLink size={18} className="group-hover:translate-x-1 transition" />
                        </a>
                      ))}
                    </div>
                  </div>

                  {/* Ethnic Wear Shopping */}
                  <div className="mb-8">
                    <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                      <span className="text-2xl">{gender === 'male' ? '🕉️' : '🪔'}</span>
                      Ethnic Wear
                    </h4>
                    <div className="grid md:grid-cols-3 gap-4">
                      {getShoppingLinks(bodyData.bodyShape, skinTone?.category).ethnic.map((shop, idx) => (
                        <a
                          key={idx}
                          href={shop.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={`${shop.color} text-white p-6 rounded-xl shadow-lg hover:shadow-xl transition transform hover:-translate-y-1 flex flex-col items-center justify-center gap-3 group`}
                        >
                          <div className="text-4xl">{shop.logo}</div>
                          <div className="font-bold text-lg">{shop.name}</div>
                          <div className="text-sm opacity-90">Shop ethnic styles</div>
                          <ExternalLink size={18} className="group-hover:translate-x-1 transition" />
                        </a>
                      ))}
                    </div>
                  </div>

                  {skinTone && (
                    <>
                      <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2 mt-8">
                        <Shirt className="text-pink-600" />
                        Shop Your Perfect Colors
                      </h3>
                      <p className="text-gray-600 mb-6">Find {gender === 'male' ? 'clothing' : 'outfits'} in colors that complement your {skinToneCategories[skinTone.category].name} skin tone:</p>
                      
                      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {getShoppingLinks(bodyData.bodyShape, skinTone.category).colors.map((shop, idx) => (
                          <a
                            key={idx}
                            href={shop.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="bg-white border-2 border-gray-200 hover:border-purple-400 p-6 rounded-xl shadow hover:shadow-lg transition transform hover:-translate-y-1 flex flex-col items-center justify-center gap-3 group"
                          >
                            <div className="text-3xl">{shop.logo}</div>
                            <div className="font-bold text-gray-800">{shop.name}</div>
                            <div className="text-sm text-gray-600 text-center">Perfect colors for you</div>
                            <ExternalLink size={16} className="text-purple-600 group-hover:translate-x-1 transition" />
                          </a>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      )}
    </div>
  );
};

export default StyleAdvisor;