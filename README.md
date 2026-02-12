### Central Retinal Vessel Segmentation for Spontaneous Venous Pulsations (SVP) Analysis 

Spontaneous Retinal Venous Pulsation (SVP) refers to the widening and narrowing of retinal veins at the optic disc. These rythmic pulsations are linked to the heartbeat and influenced by pressure changes between the eye and the brain. The absence of SVP is significant as it is related to an increased intracranial pressure (ICP) or glaucoma. This work focuses on the segmentation of the retinal vessels most indicative of a presence or absence of SVP. 

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
