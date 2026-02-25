import streamlit as st
import pandas as pd
from pathlib import Path
import base64
from fragnet.vizualize.viz import FragNetVizApp
from fragnet.vizualize.model import FragNetPreTrainViz
from streamlit_ketcher import st_ketcher
from fragnet.vizualize.model_attr import get_attr_image
# Initial page config

st.set_page_config(
     page_title='FragNet Vizualize',
     layout="wide",
     initial_sidebar_state="expanded",
)

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# sidebar
def input_callback():
    st.session_state.input = st.session_state.my_input
# def cs_sidebar():

def predict_cdrp(smiles, cell_line, cell_line_df):
    gene_expr = cell_line_df.loc[cell_line,:].values
    viz.calc_weights_cdrp(smiles, gene_expr)
    prop_prediction = -1   
    return viz, prop_prediction     



def resolve_prop_model(prop_type):

    if prop_type == 'Solubility':
        model_config = './fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/config_exp100.yaml'
        chkpt_path = './fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt'
        # model_config = './fragnet/exps/ft/pnnl_set2/fragnet_hpdl_exp1s_h4pt4_10/config_exp100.yaml'
        # chkpt_path = '../fragnet/exps/ft/pnnl_set2/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt'
        
        viz = FragNetVizApp(model_config, chkpt_path)

        prop_prediction = viz.calc_weights(selected)


    elif prop_type == 'Lipophilicity':
        model_config =  './fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/config_exp100.yaml'
        chkpt_path = './fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/ft_100.pt'
        viz = FragNetVizApp(model_config, chkpt_path)

        prop_prediction = viz.calc_weights(selected)  

    elif prop_type == 'Energy':
        model_config = '../fragnet/fragnet/exps/pt/unimol_exp1s4/config.yaml'
        chkpt_path = '../fragnet/fragnet/exps/pt/unimol_exp1s4/pt.pt'
        viz = FragNetVizApp(model_config, chkpt_path, 'energy')
        prop_prediction = viz.calc_weights(selected)

    return viz, prop_prediction, model_config, chkpt_path

def resolve_DRP(smiles, cell_line, cell_line_df):

    model_config = '../fragnet/fragnet/exps/ft/gdsc/fragnet_hpdl_exp1s_pt4_30/config_exp100.yaml'
    chkpt_path = '../fragnet/fragnet/exps/ft/gdsc/fragnet_hpdl_exp1s_pt4_30/ft_100.pt'
    viz = FragNetVizApp(model_config, chkpt_path,'cdrp')

    # viz, prop_prediction = predict_cdrp(smiles=selected, cell_line=cell_line, cell_line_df=cell_line_df)
    gene_expr = cell_line_df.loc[cell_line,:].values
    viz.calc_weights_cdrp(smiles, gene_expr)
    prop_prediction = -1   

    return viz, prop_prediction, model_config, chkpt_path


# st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=32 height=32>](https://streamlit.io/)'''.format(img_to_bytes("logomark_website.png")), unsafe_allow_html=True)
st.sidebar.header('FragNet Vizualize')

prop_type = st.sidebar.radio(
    "Select the Property type",
    # ["Solubility", "Lipophilicity", "Energy", "DRP"],
    ["Solubility", "Lipophilicity"],
    # captions = ["In logS units", "Lipophilicity", "Energy", "Drug Response Prediction"]
    captions = ["In logS units", "Lipophilicity"]
)

        # def input_callback():
        #     st.session_state.input = st.session_state.my_input
        # selected = st.text_input("Input Your Own SMILES :", key="my_input",on_change=input_callback,args=None)

selected = st.sidebar.text_input("Input SMILES :", key="my_input",on_change=input_callback,args=None,
                                 value="CC1(C)CC(O)CC(C)(C)N1[O]")
selected = st_ketcher( selected )

# if prop_type=="DRP":

#     cell_line = st.sidebar.selectbox(
#     'Select the cell line identifier',
#     ['DATA.906826',
#     'DATA.687983',
#     'DATA.910927',
#     'DATA.1240138',
#     'DATA.1240139',
#     'DATA.906792',
#     'DATA.910688',
#     'DATA.1240135',
#     'DATA.1290812',
#     'DATA.907045',
#     'DATA.906861',
#     'DATA.906830',
#     'DATA.909750',
#     'DATA.1240137',
#     'DATA.753552',
#     'DATA.907065',
#     'DATA.925338',
#     'DATA.1290809',
#     'DATA.949158',
#     'DATA.924110'])
# cell_line='DATA.924110'
# cell_line_df = pd.read_csv('../fragnet/fragnet/assets/cell_line_data.csv', index_col=0)

#     st.sidebar.write(f'selected cell line: {cell_line}')

col1, col2, col3 = st.columns(3)

if prop_type in ["Solubility", "Lipophilicity", "Energy"]:
    viz, prop_prediction, model_config, chkpt_path = resolve_prop_model(prop_type)
# elif prop_type == "DRP":
#     viz, prop_prediction, model_config, chkpt_path = resolve_DRP(selected, cell_line, cell_line_df)


hide_bond_weights = st.sidebar.checkbox("Hide bond weights")
hide_atom_weights = st.sidebar.checkbox("Hide atom weights")

with col1:
    # root='/Users/pana982/projects/esmi/models/fragnet/fragnet/'

    png, atom_weights = viz.vizualize_atom_weights(hide_bond_weights, hide_atom_weights)
    col1.image(png, caption='Atom Weights')

    # png_attr = get_attr_image(selected)
    # col1.image(png_attr, caption='Fragment Attributions') 

    attn_atoms = pd.DataFrame(atom_weights)
    attn_atoms.index.rename('Atom Index', inplace=True)
    attn_atoms.columns = ['Atom Weights']
    col1.dataframe(attn_atoms)





if prop_type == "Solubility":
    st.sidebar.write(f"Predicted Solubility (logS): {prop_prediction:.4f}")
if prop_type == "Lipophilicity":
    st.sidebar.write(f"Predicted Lipophilicity: {prop_prediction:.4f}")
if prop_type == "Energy":
    st.sidebar.write(f"Predicted Energy: {prop_prediction:.4f}")

hide_bond_weights=False
png_frag_attn, png_frag_highlight, frag_w, connection_w, atoms_in_frags = viz.frag_weight_highlight()

with col2:
    
    png_attr = get_attr_image(selected, model_config, chkpt_path)
    col2.image(png_attr, caption='Fragment Attributions') 


    col2.image(png_frag_highlight, caption='Fragments')
    st.write("Atoms in each Fragment")
    df_atoms_in_frags = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in atoms_in_frags.items() ])).T
    df_atoms_in_frags.index.rename('Fragment', inplace=True)
    st.dataframe(df_atoms_in_frags)

with col3:
    

    st.image(png_frag_attn, caption='Fragment Weights')


    st.write('Fragment Weight Values')
    st.dataframe(frag_w)

    st.write('Fragment Connection Weight Values')
    st.dataframe(connection_w)

# Add contribution analysis section
st.divider()
st.header("Contribution Analysis")

show_contributions = st.checkbox("Show Detailed Contribution Analysis", value=False)

if show_contributions:
    try:
        with st.spinner("Calculating contributions..."):
            df_atom_contrib, df_bond_contrib, df_fbond_contrib = viz.get_all_contributions(prop_type)
        
        contrib_col1, contrib_col2, contrib_col3 = st.columns(3)
        
        with contrib_col1:
            st.subheader("Atom Contributions")
            st.write("Impact of each atom on the prediction (masking approach)")
            # Sort by absolute attribution value
            df_atom_display = df_atom_contrib.copy()
            df_atom_display['abs_attr'] = df_atom_display['attr'].abs()
            df_atom_display = df_atom_display.sort_values('abs_attr', ascending=False).drop('abs_attr', axis=1)
            st.dataframe(df_atom_display, hide_index=True)
            
            # Show summary statistics
            st.write(f"**Mean Contribution:** {df_atom_contrib['attr'].mean():.4f}")
            st.write(f"**Max Contribution:** {df_atom_contrib['attr'].max():.4f} (Atom {df_atom_contrib.loc[df_atom_contrib['attr'].idxmax(), 'atom_index']})")
            st.write(f"**Min Contribution:** {df_atom_contrib['attr'].min():.4f} (Atom {df_atom_contrib.loc[df_atom_contrib['attr'].idxmin(), 'atom_index']})")
        
        with contrib_col2:
            st.subheader("Bond Contributions")
            st.write("Impact of each bond on the prediction (masking approach)")
            df_bond_display = df_bond_contrib.copy()
            df_bond_display['abs_attr'] = df_bond_display['attr'].abs()
            df_bond_display = df_bond_display.sort_values('abs_attr', ascending=False).drop('abs_attr', axis=1)
            st.dataframe(df_bond_display, hide_index=True)
            
            # Show summary statistics
            st.write(f"**Mean Contribution:** {df_bond_contrib['attr'].mean():.4f}")
            if not df_bond_contrib.empty:
                st.write(f"**Max Contribution:** {df_bond_contrib['attr'].max():.4f} (Bond {df_bond_contrib.loc[df_bond_contrib['attr'].idxmax(), 'bond_index']})")
                st.write(f"**Min Contribution:** {df_bond_contrib['attr'].min():.4f} (Bond {df_bond_contrib.loc[df_bond_contrib['attr'].idxmin(), 'bond_index']})")
        
        with contrib_col3:
            st.subheader("Fragment Bond Contributions")
            st.write("Impact of fragment connections on the prediction")
            if not df_fbond_contrib.empty:
                df_fbond_display = df_fbond_contrib.copy()
                df_fbond_display['abs_attr'] = df_fbond_display['attr'].abs()
                df_fbond_display = df_fbond_display.sort_values('abs_attr', ascending=False).drop('abs_attr', axis=1)
                st.dataframe(df_fbond_display, hide_index=True)
                
                # Show summary statistics
                st.write(f"**Mean Contribution:** {df_fbond_contrib['attr'].mean():.4f}")
                st.write(f"**Max Contribution:** {df_fbond_contrib['attr'].max():.4f}")
                st.write(f"**Min Contribution:** {df_fbond_contrib['attr'].min():.4f}")
            else:
                st.info("No fragment bonds in this molecule (single fragment)")
                
    except Exception as e:
        st.error(f"Error calculating contributions: {str(e)}")

st.sidebar.write(f"Current smiles: {selected}")