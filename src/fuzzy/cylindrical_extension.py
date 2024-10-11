import numpy as np
import matplotlib.pyplot as plt


# Triangular membership function vectorized
def triangular(x, a, b, c):
    return np.fmax(np.fmin((x - a) / (b - a), (c - x) / (c - b)), 0)

def main():
    # Initialize X the same between 0 and 1
    X = np.linspace(0, 1, 100)
    # Define two triangular membership functions as lambda functions
    mu1 = lambda x: triangular(x, 0.1, 0.3, 0.5)
    mu2 = lambda x: triangular(x, 0.4, 0.6, 0.8)
    print(max(mu1(X)))
    #
    plt.plot(X, mu1(X), label="Triangular")
    plt.plot(X, mu2(X), label="Triangular")
    # Create mesh
    X, Y = np.meshgrid(X, X)
    # Compute t-norm
    Z = np.fmin(mu1(X), mu2(Y))
    # Plot the result as a surface
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    # Other cmap options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ax.plot_surface(X, Y, Z, cmap='jet')
    # Show wireframe
    ax.plot_wireframe(X, Y, Z, color='black', rstride=2, cstride=2, alpha=0.5)
    # Show other membership functions as well (with different colors, green and red colors)
    ax.plot_surface(X, Y, mu1(X), cmap='Reds', alpha=0.2)
    ax.plot_surface(X, Y, mu2(Y), cmap='Greens', alpha=0.2)


    plt.show()


if __name__ == "__main__":
    main()
