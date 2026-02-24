### Central Retinal Vessel Segmentation for Spontaneous Venous Pulsations (SVP) Analysis 

According to the Centers for Disease Control and Prevention (CDC), approximately 80 million people worldwide are currently living with glaucoma, a number projected to exceed 111 million by 2040. In addition, elevated intracranial pressure (ICP) contributes to approximately 50,000–69,000 deaths annually in the United States, as reported by the National Institutes of Health (NIH).
A clinically significant indicator associated with both glaucoma and elevated ICP is the absence of Spontaneous Venous Pulsations (SVP).

Spontaneous Venous Pulsation refers to the rhythmic narrowing and widening of the central retinal vein as it passes through the optic disc. These rythmic pulsations are linked to the heartbeat and influenced by pressure changes between the eye and the brain. The absence of SVP is significant as it is related to an increased intracranial pressure or glaucoma. 

This work focuses on the segmentation of central retinal vessels, particularly those at the optic disc that are most indicative of the presence or absence of SVP. Accurate vessel segmentation is a critical preprocessing step for automated SVP detection and quantitative analysis, enabling objective assessment of disease risk and progression.

### Retinal Image Preprocessing
Retinal blood vessels are most clearly visualized in the green channel of fundus images, as hemoglobin strongly absorbs green light, resulting in higher vessel-to-background contrast. Therefore, the green channel was extracted for further processing.

To enhance vessel visibility, Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied. CLAHE improves local contrast while limiting noise amplification, making the retinal vessels darker and more distinguishable from surrounding tissue.

<table>
  <tr>
    <td><img src="README_Images/Original Image.png" alt="Original Image" width="300"></td>
    <td><img src="README_Images/Image Green Plane.png" alt="Image Green Plane" width="300"></td>
    <td><img src="README_Images/Image CLAHE.png" alt="Image After CLAHE" width="300"></td>

  </tr>
</table>
