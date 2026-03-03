import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentación de Clientes - K-means Clustering")
        self.root.geometry("1200x700")
        
        # Variables
        self.df = None
        self.df_scaled = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        
        # Crear la interfaz
        self.setup_ui()
        
        # Cargar datos inicialmente
        self.cargar_datos()
    
    def setup_ui(self):
        """Configura todos los elementos de la interfaz gráfica"""
        
        # Frame principal dividido en izquierda (controles) y derecha (visualizaciones)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame izquierdo para controles
        left_frame = ttk.Frame(main_frame, relief="groove", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Título
        ttk.Label(left_frame, text="Configuración del Clustering", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Control para número de clústeres
        ttk.Label(left_frame, text="Número de clústeres (K):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.k_var = tk.IntVar(value=3)
        self.k_spinbox = ttk.Spinbox(left_frame, from_=2, to=8, textvariable=self.k_var, width=10)
        self.k_spinbox.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Botón para ejecutar clustering
        self.btn_cluster = ttk.Button(left_frame, text="Ejecutar K-means", 
                                      command=self.ejecutar_clustering)
        self.btn_cluster.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Área de información (resultados)
        ttk.Label(left_frame, text="Resultados:", font=('Arial', 10, 'bold')).grid(row=3, column=0, columnspan=2, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(left_frame, width=40, height=15, wrap=tk.WORD)
        self.info_text.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Frame derecho para gráficos
        right_frame = ttk.Frame(main_frame, relief="groove", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Notebook (pestañas) para diferentes visualizaciones
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña 1: Clusters
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Clusters")
        
        # Pestaña 2: Método del codo
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Método del Codo")
        
        # Pestaña 3: Perfil de Silueta
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Perfil de Silueta")
        
        # Configurar el grid para que sea responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def cargar_datos(self):
        """Carga y prepara los datos del archivo CSV"""
        try:
            # Leer el archivo CSV
            self.df = pd.read_csv('customers.csv')
            
            # Seleccionar las características para el clustering
            features = ['AverageSpend', 'AverageFrequency']
            self.X = self.df[features].copy()
            
            # Escalar los datos (importante para K-means)
            self.X_scaled = self.scaler.fit_transform(self.X)
            
            # Mostrar información inicial
            self.mostrar_info("✅ Datos cargados correctamente\n")
            self.mostrar_info(f"Total de clientes: {len(self.df)}\n")
            self.mostrar_info("\nEstadísticas descriptivas:\n")
            self.mostrar_info(str(self.X.describe()))
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron cargar los datos: {str(e)}")
    
    def ejecutar_clustering(self):
        """Ejecuta el algoritmo K-means y muestra los resultados"""
        try:
            k = self.k_var.get()
            
            # Aplicar K-means
            self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = self.kmeans_model.fit_predict(self.X_scaled)
            
            # Añadir los clusters al DataFrame original
            self.df['Cluster'] = clusters
            
            # Calcular métricas
            silueta = silhouette_score(self.X_scaled, clusters)
            
            # Calcular estadísticas por cluster
            stats = self.df.groupby('Cluster')[['AverageSpend', 'AverageFrequency']].agg(['mean', 'std', 'count'])
            
            # Mostrar resultados
            self.mostrar_resultados(k, silueta, stats)
            
            # Actualizar visualizaciones
            self.actualizar_graficos()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el clustering: {str(e)}")
    
    def mostrar_resultados(self, k, silueta, stats):
        """Muestra los resultados del clustering en el área de texto"""
        self.info_text.delete(1.0, tk.END)
        
        self.info_text.insert(tk.END, "📊 RESULTADOS DEL CLUSTERING\n")
        self.info_text.insert(tk.END, "="*40 + "\n\n")
        
        self.info_text.insert(tk.END, f"Número de clusters (K): {k}\n")
        self.info_text.insert(tk.END, f"Coeficiente de Silueta: {silueta:.4f}\n")
        
        # Interpretación del coeficiente de silueta
        self.info_text.insert(tk.END, "\n📈 Interpretación:\n")
        if silueta > 0.7:
            self.info_text.insert(tk.END, "✅ Excelente separación entre clusters\n")
        elif silueta > 0.5:
            self.info_text.insert(tk.END, "👍 Buena separación entre clusters\n")
        elif silueta > 0.3:
            self.info_text.insert(tk.END, "⚠️ Separación razonable\n")
        else:
            self.info_text.insert(tk.END, "❌ Mala separación (clusters superpuestos)\n")
        
        self.info_text.insert(tk.END, "\n" + "="*40 + "\n\n")
        self.info_text.insert(tk.END, "📊 ESTADÍSTICAS POR CLUSTER\n\n")
        self.info_text.insert(tk.END, "Centroides (valores originales):\n")
        
        # Mostrar centroides en escala original
        centroides_originales = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
        for i, centroide in enumerate(centroides_originales):
            self.info_text.insert(tk.END, f"Cluster {i}: Gasto=${centroide[0]:.2f}, Frecuencia={centroide[1]:.1f}\n")
        
        self.info_text.insert(tk.END, "\nDistribución de clientes:\n")
        for cluster in range(k):
            count = len(self.df[self.df['Cluster'] == cluster])
            self.info_text.insert(tk.END, f"Cluster {cluster}: {count} clientes ({count/len(self.df)*100:.1f}%)\n")
    
    def actualizar_graficos(self):
        """Actualiza todos los gráficos en las pestañas"""
        # Limpiar pestañas
        for widget in self.tab1.winfo_children():
            widget.destroy()
        for widget in self.tab2.winfo_children():
            widget.destroy()
        for widget in self.tab3.winfo_children():
            widget.destroy()
        
        # Gráfico 1: Clusters
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        
        # Colores para los clusters
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        # Graficar puntos por cluster
        for cluster in range(self.k_var.get()):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            ax1.scatter(cluster_data['AverageSpend'], 
                       cluster_data['AverageFrequency'],
                       c=colors[cluster % len(colors)],
                       label=f'Cluster {cluster}',
                       alpha=0.6,
                       edgecolors='black',
                       linewidth=0.5)
        
        # Graficar centroides
        centroides_originales = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
        ax1.scatter(centroides_originales[:, 0], 
                   centroides_originales[:, 1],
                   c='yellow',
                   marker='X',
                   s=200,
                   edgecolors='black',
                   linewidth=2,
                   label='Centroides')
        
        ax1.set_xlabel('Gasto Promedio ($)')
        ax1.set_ylabel('Frecuencia Promedio')
        ax1.set_title('Segmentación de Clientes por K-means')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        canvas1 = FigureCanvasTkAgg(fig1, self.tab1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico 2: Método del codo
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        inertias = []
        K_range = range(1, 9)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
        
        ax2.plot(K_range, inertias, 'bo-')
        ax2.set_xlabel('Número de Clusters (K)')
        ax2.set_ylabel('Inercia')
        ax2.set_title('Método del Codo para determinar K óptimo')
        ax2.grid(True, alpha=0.3)
        
        # Marcar el K seleccionado
        ax2.axvline(x=self.k_var.get(), color='red', linestyle='--', alpha=0.7, 
                   label=f'K seleccionado = {self.k_var.get()}')
        ax2.legend()
        
        canvas2 = FigureCanvasTkAgg(fig2, self.tab2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico 3: Perfil de Silueta
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        
        # Calcular silueta para cada punto
        silhouette_vals = silhouette_samples(self.X_scaled, self.df['Cluster'])
        
        y_lower = 10
        for i in range(self.k_var.get()):
            # Agregar silhouette scores del cluster i
            cluster_silhouette_vals = silhouette_vals[self.df['Cluster'] == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = len(cluster_silhouette_vals)
            y_upper = y_lower + size_cluster_i
            
            color = colors[i % len(colors)]
            ax3.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            # Etiqueta del cluster
            ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 5
        
        ax3.axvline(x=silhouette_score(self.X_scaled, self.df['Cluster']), 
                   color="red", linestyle="--", linewidth=2, label='Silueta promedio')
        ax3.set_xlabel("Coeficiente de Silueta")
        ax3.set_ylabel("Cluster")
        ax3.set_title("Perfil de Silueta para cada Cluster")
        ax3.legend()
        
        canvas3 = FigureCanvasTkAgg(fig3, self.tab3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def mostrar_info(self, texto):
        """Muestra información en el área de texto"""
        self.info_text.insert(tk.END, texto)
        self.info_text.see(tk.END)

def main():
    """Función principal para ejecutar la aplicación"""
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()