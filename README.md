# Implementation of Extended Kalman Filter in C++ using Unity Extended-Kalman-Filter Simulator , Simulated lidar and radar measurements readings detecting a bicycle travelling around the car in the simulation environment to track the bicycle's position and velocity.

# Problem Statement Description
For this project the task is to implement Extended Kalman Filter in C++ using Unity Extended-Kalman-Filter Simulator , simulated lidar and radar measurements readings detecting a bicycle travelling around the simulated car in the simulation environment to track the bicycle's position and velocity. Lidar measurements are red circles, radar measurements are blue circles with an arrow pointing in the direction of the observed angle, and estimation markers are green triangles.The simulator provides the script the measured data (either lidar or radar), and the script feeds back the measured estimation marker, and RMSE values from its Kalman filter.

# Solution Criteria
px, py, vx and vy RMSE (Root-Mean-Square-Estimate) values should be less than or equal to the values [.11, .11, 0.52, 0.52]
respectively.

# Results Showcase
<table>
  <tr>
    <td colspan="3" align="center">RESULTS</td>
  </tr>
  <tr>
    <td> </td>
    <td>Zoomed In View</td>
    <td>Zoomed Out View</td>
  </tr>
  <tr>
    <td>Dataset 1</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604926-027b3e00-4a6e-11e9-92a6-974434942301.gif"></td></tr><tr>
      <td>LINK</td>     
      <td>https://youtu.be/LJ9kPpbqWFk</td></tr></table></td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604913-fa230300-4a6d-11e9-8ece-a657faaf7c27.gif"></td></tr><tr>
      <td>LINK</td><td>https://youtu.be/0U46mhdeEpg</td></tr></table></td>
  </tr>
  <tr>
    <td>Dataset 2</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604956-0dce6980-4a6e-11e9-9d34-c442ab505fd2.gif"></td></tr><tr>
      <td>LINK</td><td>https://youtu.be/LhkD237zxQQ</td></tr></table></td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604944-07d88880-4a6e-11e9-90f4-f3b854858209.gif"></td></tr><tr>
      <td>LINK</td><td>https://youtu.be/1hsIQAJ0aRY</td></tr></table></td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="3" align="center">FINAL OUTPUT SCREENSSHOT</td>
  </tr>
  <tr>
    <td> </td>
    <td>Zoomed In View</td>
    <td>Zoomed Out View</td>
  </tr>
  <tr>
    <td>Dataset 1</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605688-d9f44380-4a6f-11e9-8f12-45ac17ada9b2.png"></td></tr></table>
      </td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605706-de206100-4a6f-11e9-818d-fbb33574dfb5.png"></td></tr></table></td>
  </tr>
  <tr>
    <td>Dataset 2</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605713-e1b3e800-4a6f-11e9-94a0-7ea8ef0f631d.png"></td></tr>
      </table></td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605718-e6789c00-4a6f-11e9-84ba-7e7778b3a8fe.png"></td></tr>
      </table></td>
  </tr>
</table>

