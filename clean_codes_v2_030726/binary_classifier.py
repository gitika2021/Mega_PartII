import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from skimage import measure
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from skimage.util import pad
from functools import partial
import time
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import seaborn as sns

class FitCircleTo2DBinaryImage():

    def __init__(self, image_data_2d, threshold = None, verbose=False, deviation_estimator='flux'):
      self.image = image_data_2d
      self.sigma_bgr = 10
      self.threshold = threshold
      self.verbose = verbose
      self.deviation_estimator = deviation_estimator  # 'flux' or 'area'


    def pad_image(self, pad_width_pixels = 10, pad_value = 0):
      self.image_padded= np.pad(self.image, pad_width=pad_width_pixels, mode='constant', constant_values=pad_value)
      # self.image_padded[self.binary_padded] = 1
      # self.image_padded[~self.binary_padded] = 0

      # --- Image center ---
      self.H, self.W = self.image_padded.shape
      self.x0, self.y0 = self.W / 2, self.H / 2

      self.x = np.arange(self.H)
      self.y = np.arange(self.W )
      self.X, self.Y = np.meshgrid(self.x, self.y)

      # Create coordinate grids
      self.y_indices, self.x_indices = np.indices((self.H, self.W))
      # Calculate distance from center for each pixel in the full image
      self.dist_from_center = np.sqrt((self.x_indices - self.x0)**2 + (self.y_indices - self.y0)**2)
      # print('self.y_indices, self.x_indices',self.y_indices, self.x_indices)

    def bin_flux_value(self):
      # Create histogram (you can set number of bins)
      counts, bin_edges = np.histogram(self.image_padded.flatten(), bins=100)

      # Compute bin centers from edges
      self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
      #print('bin_centers',bin_centers)

      # Plot using bin centers
      plt.plot(bin_centers, counts, drawstyle='steps-mid', marker='o')
      plt.xlabel("Value")
      plt.ylabel("Count")
      plt.title("1D Histogram with Bin Centers")
      plt.grid(True)
      plt.show()


    def background_from_flux_enclosure(self, flux_ratio_cut = 0.9973, show=True ):
      """
      Get background level by calculating total flux area contained within the contour such that the total flux within the contour is >= flux_ratio_cut
      Set flux_ratio_cut = area enclosed corresponds some n sigma value.
      E.g. To find the contour corresponding to sigma level set :
          1 sigma level ==> flux_ratio_cut = 0.6827; ~68% of total flux is within this region
          2 sigma level ==> flux_ratio_cut = 0.9545; ~95% of total flux is within this region
          3 sigma level ==> flux_ratio_cut = 0.9973; ~99% of total flux is within this region
      """


      # # Integrate intensity over entire image
      # I_y = integrate.simpson(self.image_padded, self.y, axis=0)
      # I_total = integrate.simpson(I_y, self.x)
      I_total = np.nansum(self.image_padded)
      # print('I_total',I_total)

      img_max_val = self.image_padded.max()
      img_min_val = self.image_padded.min()
      # print('img_max_val,img_min_val',img_max_val,img_min_val)

      # choose flux levels (or probabilities) bins
      # self.bin_flux_value()
      # bins_val = self.bin_centers
      bins_val = np.logspace(-0.04, -5, num=100) # np.linspace(img_max_val,img_min_val,40)

      # find the total area contained within this contour value and stop when it reaches flux_ratio_cut
      for val in bins_val:

        # # method 1: integrating over circular regions
        # rows, cols = np.where(self.image_padded >= val)
        # distances = np.sqrt((rows - self.x0)**2 + (cols - self.y0)**2)          # radius of each pixel from the center
        # radius_mask = distances.max()
        # print('radius_mask', radius_mask)                                       # find the max radius corresponding to val

        # # choose region only within the radius_mask
        # mask_region = self.dist_from_center <= radius_mask #self.dist_from_center <= radius_mask**2
        # I_y = integrate.simpson(image_masked_region, self.y, axis=0)
        # I_total_sel = integrate.simpson(I_y, self.x)

        # method 2: integrating over contour values (default method)
        mask_region = self.image_padded > val
        image_masked_region = np.where(mask_region, self.image_padded, np.nan)       # set pixels outside the mask to 0  Z_masked_region = np.where(mask_region, Z, 0)
        I_total_sel = np.nansum(image_masked_region)

        if show==True:
          plt.imshow(image_masked_region, cmap='gray')
          plt.show()


        area_enclosed = I_total_sel/I_total
        # print(f"Total integral over selcted region: {I_total_sel:.5f}")

        if area_enclosed > flux_ratio_cut:
          if self.verbose:
            print(f"Condition met: area enclosed has reached {area_enclosed} at flux value {val}")
          # print(f"Condition met: area enclosed has reached {area_enclosed} at flux value {val} for radius {radius_mask} when center is at {self.x0,self.y0}")
          self.threshold = val
          # print('self.threshold',self.threshold)
          if show==True:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.image_padded, cmap='gray')
            contour = plt.contour(self.image_padded, levels=[self.threshold], colors='red', linewidths=2)
            ax.set_title(f"probability threshold {self.threshold}", fontsize=10)
            ax.legend(loc='lower right', fontsize=8)
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.title("Region Integrated (Inside Circle)")
            plt.show()

          break


    def extract_boundary_contour(self,):
      self.contours = measure.find_contours(self.image_padded, level=self.threshold)
      #binary_padded = self.image_padded > threshold
      #contours = measure.find_contours(binary_padded.astype(float), level=0.5)
      #print('contours.shape',contours)

      if not self.contours:
          raise ValueError("No contours found at 3σ level.")
      self.contour = max(self.contours, key=len)

      # --- Extract coordinates of contour ---
      self.x_contour = self.contour[:, 1]
      self.y_contour = self.contour[:, 0]

      # --- Get intensity (signal) values at contour points ---
      x_idx = np.clip(self.x_contour.astype(int), 0, self.W - 1)
      y_idx = np.clip(self.y_contour.astype(int), 0, self.H - 1)
      signal_weights = self.image_padded[y_idx, x_idx]  # Brightness at contour points

      # Normalize weights
      self.weights = 1#signal_weights / np.max(signal_weights)

      # Compute radius and angle of each contour point
      self.r_contour = np.sqrt((self.x_contour - self.x0)**2 + (self.y_contour - self.y0)**2)
      angles = np.arctan2(self.y_contour - self.y0, self.x_contour - self.x0)
      angles = np.mod(angles, 2 * np.pi)  # Ensure 0 to 2pi



    # def fit_circle(self,):
    #    #r_contour0 = self. r_contour
    #   def weighted_residual_squared_sum(r):
    #     # --- Weighted least squares objective ---
    #     residuals = self.r_contour - r
    #     return np.mean(self.weights * residuals**2)


    #   # --- Minimize weighted squared error ---
    #   res = minimize_scalar(weighted_residual_squared_sum, bounds=(5, min(self.H, self.W)/2), method='bounded')

    #   self.best_radius = res.x
    #   self.residuals = self.r_contour - self.best_radius

    #   n_data_points = len(self.r_contour)#total number of independent data points used in fitting
    #   n_params = 1 #number of fitted parameters (e.g., 1 if radius only)

    #   chi2_min = res.fun
    #   dof = n_data_points - n_params
    #   self.p_value = chi2.sf(chi2_min, dof)

    # def get_fitted_circle(self):
    #   # Fitted circle
    #   theta = np.linspace(0, 2*np.pi, 500)
    #   self.x_circle = self.x0 + self.best_radius * np.cos(theta)
    #   self.y_circle = self.y0 + self.best_radius * np.sin(theta)

    def fit_circle(self):
      """
      Fit a circle (cx, cy, r) to contour points using weighted least squares.
      """

      # --- Extract variables ---
      x = self.x_contour
      y = self.y_contour
      w = self.weights
      N = len(x)

      # --- Define residual function ---
      def objective(params):
          cx, cy, r = params
          # Distance of points from proposed center
          d = np.sqrt((x - cx)**2 + (y - cy)**2)
          residuals = d - r
          return np.mean(w * residuals**2)

      # --- Initial guess from image center + mean radius ---
      cx0 = self.W / 2
      cy0 = self.H / 2
      r0 = np.mean(np.sqrt((x - cx0)**2 + (y - cy0)**2))

      initial_guess = np.array([cx0, cy0, r0])

      # --- Boundaries ---
      bounds = [
          (0, self.W),              # cx
          (0, self.H),              # cy
          (5, min(self.H, self.W))  # radius
      ]

      # --- Minimize ---
      res = minimize(lambda p: objective(p),
                    x0=initial_guess,
                    bounds=bounds,
                    method='L-BFGS-B')

      # --- Save best-fit parameters ---
      cx_best, cy_best, r_best = res.x
      self.best_cx = cx_best
      self.best_cy = cy_best
      self.best_radius = r_best

      # --- Residuals ---
      d = np.sqrt((x - cx_best)**2 + (y - cy_best)**2)
      self.residuals = d - r_best

      # --- Chi-square statistics ---
      chi2_min = np.sum(w * self.residuals**2)
      dof = N - 3   # 3 fitted parameters: cx, cy, r
      self.p_value = chi2.sf(chi2_min, dof)



    def get_fitted_circle(self):
      # Fitted circle
      theta = np.linspace(0, 2*np.pi, 500)
      self.x_circle = self.best_cx + self.best_radius * np.cos(theta)
      self.y_circle = self.best_cy + self.best_radius * np.sin(theta)

      self.y_indices, self.x_indices = np.indices((self.H, self.W))
      # Calculate distance from center for each pixel in the full image
      self.dist_from_center = np.sqrt((self.x_indices - self.best_cx)**2 + (self.y_indices - self.best_cy)**2)

    def plot_fitted_circle(self,):
      import matplotlib as mpl

      mpl.rcParams.update({
          "font.size": 15,
          "axes.labelsize": 15,
          "axes.titlesize": 15,
          "xtick.labelsize": 15,
          "ytick.labelsize": 15,
          "legend.fontsize": 15,
          "figure.titlesize": 15,
      })

      # --- Plot image with overlaid contour residuals ---
      fig, ax = plt.subplots(figsize=(6, 6))
      img = ax.imshow(self.image_padded, cmap='inferno')
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_xlabel("")   # ensure no leftover labels
      ax.set_ylabel("")

      divider = make_axes_locatable(ax)
      cax = divider.append_axes("top", size="5%", pad=0)
      ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
      cbar = plt.colorbar(img, cax=cax, orientation='horizontal', ticks=ticks, label ='Probability')
      cbar.ax.xaxis.set_ticks_position('top')
      cbar.ax.xaxis.set_label_position('top')
      cbar.ax.tick_params(pad=2)

      cbar.ax.margins(x=0.02)      # add a tiny margin on both ends
      cbar.ax.set_xlim(0, 1)   



      # Plot contour points with color mapped to residual magnitude
      sc = ax.scatter(self.x_contour, self.y_contour, c=self.residuals, cmap='BrBG_r', s=20, marker='o', edgecolors='k', linewidths=0.2)
      #cb = plt.colorbar(sc, ax=ax, label='Residual (r_contour - r_fit)')
      #cb = plt.colorbar(sc, ax=ax, fraction=0.048, pad=0.0, label='Residual (r_contour - r_fit)')
      #cb.ax.tick_params(labelsize=8)

      cax2 = divider.append_axes("right", size="5%", pad=0)
      ticks = np.linspace(int(self.residuals.min()),int(self.residuals.max()),5)
      cbar2 = plt.colorbar(sc, cax=cax2, orientation='vertical', ticks=ticks, label='Distance (r_contour - r_fit)')
      #cbar2.ax.xaxis.set_ticks_position('right')
      #cbar2.ax.xaxis.set_label_position('right')
      cbar2.ax.tick_params(pad=2)


      # Optional: overlay fitted circle for reference
      #ax.plot(self.x_circle, self.y_circle, 'blue', label=f'Fitted Circle (r={self.best_radius:.2f})', linewidth=1)
      ax.plot(self.x_circle, self.y_circle, 'cyan', label=f'Fitted Circle', linewidth=1.5)
      #ax.scatter(x_circle, y_circle, 'lime', label='Fitted Circle')
      ax.set_title(f"Residuals of {self.sigma_bgr}σ Contour from Fitted Circle", fontsize=10)
      ax.legend(loc='lower right', fontsize=8)
      ax.set_aspect('equal')
      plt.tight_layout()
      plt.savefig('megapart1_circle_overlay.pdf')
      plt.show()


    def extract_regions(self,):
      self.binary_padded =  self.image_padded > self.threshold

      # Region Selected: This is the shape that is selected for fitting
      self.masked_image = np.where(self.binary_padded, self.image_padded, np.nan) # masked_image is shape extracted and rest pixels set to nan, so that pixels are transparent in imshow
      self.num_pixels_shape = np.sum(~np.isnan(self.masked_image)) # Count how many pixels are NOT NaN (i.e., included in sum)
      self.total_flux_shape = np.nansum(self.masked_image)

      # Inside Region: Total flux and number of pixels inside the circle that are above > threshold
      self.inside_mask = (self.dist_from_center <= self.best_radius) & (~np.isnan(self.masked_image)) # Create mask: pixels inside radius AND not NaN in masked_image
      self.inside_region = np.where(self.inside_mask, self.masked_image, np.nan) # Extract circular region (keep pixels inside circle, set other
      self.num_pixels_inside = np.sum(self.inside_mask)
      self.total_flux_inside = np.nansum(self.inside_region)

      # Outside Region: Total flux and number of pixels outside the circle above chosen threshold
      self.outside_mask = (~self.inside_mask) & (~np.isnan(self.masked_image)) # circle_mask is True inside circle & valid pixels
      self.outside_region = np.where(self.outside_mask, self.masked_image, np.nan)  # Extract outside region (keep pixels outside circle & valid, else NaN)
      self.num_pixels_outside = np.sum(self.outside_mask)
      self.total_flux_outside = np.nansum(self.outside_region)

      # Circular Region: Total flux and number of pixels within the best fit circle
      self.circle_mask = self.dist_from_center <= self.best_radius
      self.circular_region = np.where(self.circle_mask, self.image_padded, np.nan)
      self.num_pixels_circle = np.sum(self.circle_mask ) #inside_circle_pixels.size
      self.total_flux_circle = np.nansum(self.circular_region)


      self.total_flux_image = np.nansum(self.image_padded)
      self.total_flux_outside_image = self.total_flux_image - self.total_flux_circle

      if self.verbose:
        print('total_flux_shape, num_pixels_tot',self.total_flux_shape, self.num_pixels_shape)
        print('total_flux_inside, num_pixels_inside', self.total_flux_inside, self.num_pixels_inside)
        print('total_flux_outside, num_pixels_outside', self.total_flux_outside, self.num_pixels_outside)
        print('total_flux_circle, num_pixels_circle', self.total_flux_circle, self.num_pixels_circle)
        print('total_flux_outside_image',self.total_flux_outside_image)
        print('\n')


    def plot_extracted_regions(self,):
      cmap='viridis'

      fig, axs = plt.subplots(1, 1, figsize=(3, 3))
      axs.set_xticks([])
      axs.set_yticks([])
      axs.tick_params(bottom=False, left=False) # Optionally also remove tick lines if any remain (edge case)

      im1 = axs.imshow(self.image_padded, cmap=cmap)
      plt.colorbar(im1, ax=axs, ticks=[0.3, 0.6, 0.9], fraction=0.048, pad=0.0)
      #axs.plot(self.x_circle, self.y_circle, 'red', label=f'Fitted Circle (r={self.best_radius:.2f})', linewidth=1)
      axs.set_title('Prediction Map', fontsize=8)
      #axs[0].legend(fontsize=8)


      vmin = im1.get_clim()[0]
      vmax = im1.get_clim()[1]
      cmap = im1.get_cmap()

      plt.tight_layout()
      plt.savefig('org_image.pdf')
      plt.show()


      fig, axs = plt.subplots(1, 4, figsize=(12, 8))

      #axs[0].axis('off')
      axs[0].set_xticks([])
      axs[0].set_yticks([])
      axs[0].tick_params(bottom=False, left=False) # Optionally also remove tick lines if any remain (edge case)

      axs[1].set_xticks([])
      axs[1].set_yticks([])
      axs[1].tick_params(bottom=False, left=False) # Optionally also remove tick lines if any remain (edge case)

      axs[2].set_xticks([])
      axs[2].set_yticks([])
      axs[2].tick_params(bottom=False, left=False) # Optionally also remove tick lines if any remain (edge case)

      axs[3].set_xticks([])
      axs[3].set_yticks([])
      axs[3].tick_params(bottom=False, left=False) # Optionally also remove tick lines if any remain (edge case)


      im1 = axs[0].imshow(self.masked_image, cmap=cmap)
      #plt.colorbar(im1, ax=axs[0], ticks=[0.3, 0.6, 0.9], fraction=0.046, pad=0.01)
      plt.colorbar(im1, ax=axs[0], ticks=[0.3, 0.6, 0.9], fraction=0.048, pad=0.0)
      axs[0].plot(self.x_circle, self.y_circle, 'red', label=f'Fitted Circle (r={self.best_radius:.2f})', linewidth=1)
      axs[0].set_title('Region Selected', fontsize=8)
      #axs[0].legend(fontsize=8)


      # vmin = im1.get_clim()[0]
      # vmax = im1.get_clim()[1]
      # cmap = im1.get_cmap()


      axs[1].imshow(self.inside_region, cmap=cmap, vmin=vmin, vmax=vmax)
      axs[1].plot(self.x_circle, self.y_circle, 'red', label=f'Fitted Circle (r={self.best_radius:.2f})', linewidth=1)
      axs[1].set_title('Region Inside Circle', fontsize=8)


      axs[2].imshow(self.outside_region, cmap=cmap, vmin=vmin, vmax=vmax)
      axs[2].plot(self.x_circle, self.y_circle, 'red', label=f'Fitted Circle (r={self.best_radius:.2f})', linewidth=1)
      axs[2].set_title('Region Outside Circle', fontsize=8)


      axs[3].imshow(self.circular_region, cmap=cmap, vmin=vmin, vmax=vmax)
      axs[3].plot(self.x_circle, self.y_circle, 'red', label=f'Fitted Circle (r={self.best_radius:.2f})', linewidth=1)
      axs[3].set_title('Region Within Circle', fontsize=8)

      plt.tight_layout()
      plt.savefig('circle_decision.pdf')
      plt.show()

    def find_deviation_from_circle(self,):
      """
      For perfect circles: 1. self.ratio_flx_inside_to_circle ~ 1 AND self.ratio_flx_outside_to_circle ~ 0
                           2. self.ratio_num_inside_to_circle ~ 1 AMD self.ratio_num_outside_to_circle ~ 0
                           3. self.ratio_avg_squared_dist
      """
      self.ratio_flx_inside_to_circle = self.total_flux_inside/self.total_flux_circle
      self.ratio_flx_outside_to_circle = self.total_flux_outside/self.total_flux_circle

      self.ratio_num_inside_to_circle = self.num_pixels_inside/self.num_pixels_circle
      self.ratio_num_outside_to_circle = self.num_pixels_outside/self.num_pixels_circle

      self.ratio_excess_flux_within_circle = (self.total_flux_circle - self.total_flux_outside_image)/self.total_flux_circle
      self.ratio_excess_num_within_circle = (self.num_pixels_inside - self.num_pixels_outside)/self.num_pixels_circle


      r_r0_sqr = np.mean(self.residuals**2)
      r_r0 = np.sqrt(r_r0_sqr)
      #self.ratio_avg_squared_dist = abs(np.mean(self.residuals**2)-self.best_radius**2)/(self.best_radius**2)
      self.ratio_avg_squared_dist = abs(r_r0)/(self.best_radius)
      # print('ratio_flx_inside_to_circle, ratio_num_inside_to_circle',self.ratio_flx_inside_to_circle, self.ratio_num_inside_to_circle)
      # print('ratio_num_outside_to_circle, ratio_num_outside_to_circle',self.ratio_num_outside_to_circle, self.ratio_num_outside_to_circle)
      # print('ratio_avg_squared_dist',self.ratio_avg_squared_dist)

    def predict_shape(self,show=True, cut1=0.95, cut2=0.1):
      if self.deviation_estimator == 'area':
        self.ratio_inside_to_circle = self.ratio_num_inside_to_circle
        self.ratio_outside_to_circle = self.ratio_num_outside_to_circle
        self.ratio_excess_within_circle = self.ratio_excess_num_within_circle
        self.ratio_limit = cut1#0.95 #0.90

      elif self.deviation_estimator == 'flux':
        self.ratio_inside_to_circle = self.ratio_flx_inside_to_circle
        self.ratio_outside_to_circle = self.ratio_flx_outside_to_circle
        self.ratio_excess_within_circle = self.ratio_excess_flux_within_circle
        self.ratio_limit = cut1#0.90


      #if (self.ratio_inside_to_circle < 0.9) and (self.ratio_outside_to_circle > 0.10) and (self.ratio_avg_squared_dist > 0.1) :
      #if (abs(self.ratio_inside_to_circle - self.ratio_outside_to_circle) < 0.90) and (self.ratio_avg_squared_dist > 0.1) :
      #if (self.ratio_excess_within_circle < self.ratio_limit) and (self.ratio_avg_squared_dist > 0.1) :
      if (self.ratio_excess_within_circle < self.ratio_limit) and (self.ratio_avg_squared_dist > cut2) :

        self.shape_str = 'Not Circle'
        self.shape_class = 1#0
      else:
        self.shape_str = 'Circle'
        self.shape_class = 0
      #print('hi')
      if show:
        print('show',show)
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        axs.imshow(self.image_padded, cmap='gray')
        sc = axs.scatter(self.x_contour, self.y_contour, c=self.residuals, cmap='BrBG_r', s=20, marker='o', edgecolors='k', linewidths=0.2)
        axs.plot(self.x_circle, self.y_circle, 'lime', label=f'Fitted Circle (r={self.best_radius:.2f})', linewidth=1)
        axs.set_title(f'{self.shape_str}, CI={np.round(self.ratio_excess_within_circle,2)},CII={np.round(self.ratio_avg_squared_dist,3)}', fontsize=8)
        axs.legend(fontsize=8)
        plt.tight_layout()
        plt.show()



#=======================================================================================
def process_single_image(idx, images, deviation_estimator, cut1, cut2, pad_width,show=False):

    image = images[idx]

    fitimg = FitCircleTo2DBinaryImage(image, deviation_estimator=deviation_estimator)
    fitimg.pad_image(pad_width_pixels=pad_width)
    fitimg.background_from_flux_enclosure(flux_ratio_cut=0.9545, show=False)
    fitimg.extract_boundary_contour()
    fitimg.fit_circle()
    #fitimg.extract_boundary_contour()
    fitimg.get_fitted_circle()
    fitimg.extract_regions()
    fitimg.find_deviation_from_circle()
    fitimg.predict_shape(show=show, cut1=cut1, cut2=cut2)
    #print('show',show)
    return (
        idx,
        fitimg.shape_class,
        fitimg.shape_str,
        fitimg.ratio_excess_within_circle,
        fitimg.ratio_avg_squared_dist,
        [fitimg.best_cx, fitimg.best_cy, fitimg.best_radius],
        fitimg.image_padded,
    )

def batch_predict_shape(
        images='',
        show=True,
        deviation_estimator='area',
        cut1=0.95,
        cut2=0.1,
        num_cpus=None):

    N = images.shape[0]
    nx, ny = images[0].shape
    pad_width = 10
    indices = list(range(N))

    # ===== Allocate result arrays =====
    predictions     = np.ones(N)
    predictions_str = np.empty(N, dtype=object)
    values_cutI     = np.ones(N)
    values_cutII    = np.ones(N)
    circle_fitted   = np.zeros((N, 3))
    final_maps      = np.zeros((N, nx + 2*pad_width, ny + 2*pad_width))

    # ===== Prepare the parallel task =====
    task = partial(
        process_single_image,
        images=images,
        deviation_estimator=deviation_estimator,
        cut1=cut1,
        cut2=cut2,
        pad_width=pad_width,
        show = show
    )

    print(f"Running {N} images using {num_cpus} CPUs...")

    # ===== Run in Parallel =====
    #from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        for (
            idx,
            pred,
            pred_str,
            cutI,
            cutII,
            circle,
            fmap
        ) in tqdm(executor.map(task, indices), total=N):

            predictions[idx]     = pred
            predictions_str[idx] = pred_str
            values_cutI[idx]     = cutI
            values_cutII[idx]    = cutII
            circle_fitted[idx]   = circle
            final_maps[idx]      = fmap

    # ===== Compute accuracy =====
    #n_circle = np.sum(predictions > 0)
    #accuracy = n_circle / N
    non_circle = np.sum(predictions == 1)
    accuracy = non_circle / N

    return (
        predictions,
        accuracy,
        values_cutI,
        values_cutII,
        predictions_str,
        circle_fitted,
        final_maps
    )



def plot_confusion_matrix(tp, tn, fp, fn, class_names=("Negative", "Positive"), savefig=None):
    """
    Plots a confusion matrix given TP, TN, FP, FN.
    
    Args:
        tp, tn, fp, fn: int
        class_names: tuple of (negative_class, positive_class)
    """
    # Confusion matrix layout:

    # [[FN, TN],
    #  [TP, FP]]

    cm = np.array([[fn, tn],
                   [tp, fp]])

    plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=class_names,
    #             yticklabels=class_names)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
             xticklabels=['1','0'],
            yticklabels=['0','1'])
    # sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(
            savefig,
            dpi=500,
            bbox_inches='tight',
            pad_inches=0.2
        )
                
    plt.show()
    
def estimate_ml_metrics(binary_true, binary_predicted, savefig=None):

    y_true = np.asarray(binary_true)
    y_pred = np.asarray(binary_predicted)

    # Boolean masks
    TP_mask = (y_true == 1) & (y_pred == 1)
    TN_mask = (y_true == 0) & (y_pred == 0)
    FP_mask = (y_true == 0) & (y_pred == 1)
    FN_mask = (y_true == 1) & (y_pred == 0)

    # Counts
    TP = np.sum(TP_mask)
    TN = np.sum(TN_mask)
    FP = np.sum(FP_mask)
    FN = np.sum(FN_mask)

    def safe_div(n, d):
        return n / d if d != 0 else 0.0

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    FPR = safe_div(FP, FP + TN)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(TP + TN, TP + TN + FP + FN)

    # Indices
    TP_idx = np.where(TP_mask)[0]
    TN_idx = np.where(TN_mask)[0]
    FP_idx = np.where(FP_mask)[0]
    FN_idx = np.where(FN_mask)[0]

    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)

    if savefig is not None:
        plot_confusion_matrix(TP, TN, FP, FN, class_names=("TN", "TP"), savefig=savefig)                   
    
    # print("TP:", TP, "Indices:", TP_idx)
    # print("TN:", TN, "Indices:", TN_idx)
    # print("FP:", FP, "Indices:", FP_idx)
    # print("FN:", FN, "Indices:", FN_idx)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Accuracy:", accuracy)

    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "FPR": FPR,
            "f1": f1,
        },
        "indices": {
            "TP": TP_idx,
            "TN": TN_idx,
            "FP": FP_idx,
            "FN": FN_idx,
        }
    }


