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
     page_title='FragNet Visualize',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="🧪"
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
st.sidebar.title('🧪 FragNet Visualize')
st.sidebar.markdown('---')

st.sidebar.subheader("⚙️ Configuration")
prop_type = st.sidebar.radio(
    "Property Type",
    ["Solubility", "Lipophilicity"],
    captions = ["In logS units", "Lipophilicity coefficient"],
    help="Select the molecular property to predict and visualize"
)

st.sidebar.markdown('---')

        # def input_callback():
        #     st.session_state.input = st.session_state.my_input
        # selected = st.text_input("Input Your Own SMILES :", key="my_input",on_change=input_callback,args=None)

st.sidebar.subheader("📝 Molecule Input")
selected = st.sidebar.text_input(
    "SMILES String", 
    key="my_input",
    on_change=input_callback,
    args=None,
    value="CC1(C)CC(O)CC(C)(C)N1[O]",
    help="Enter a valid SMILES string for the molecule"
)
selected = st_ketcher(selected)

st.sidebar.markdown('---')
st.sidebar.subheader("🎨 Display Options")

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

if prop_type in ["Solubility", "Lipophilicity", "Energy"]:
    viz, prop_prediction, model_config, chkpt_path = resolve_prop_model(prop_type)
# elif prop_type == "DRP":
#     viz, prop_prediction, model_config, chkpt_path = resolve_DRP(selected, cell_line, cell_line_df)

# Main title
st.title("🧬 FragNet Molecular Property Visualization")
st.markdown("**Interpretable graph neural network predictions with fragment-based analysis**")
st.markdown('---')

# Display prediction in a prominent metric card
if prop_type == "Solubility":
    st.metric(label="📊 Predicted Solubility (logS)", value=f"{prop_prediction:.4f}")
elif prop_type == "Lipophilicity":
    st.metric(label="📊 Predicted Lipophilicity", value=f"{prop_prediction:.4f}")
elif prop_type == "Energy":
    st.metric(label="📊 Predicted Energy", value=f"{prop_prediction:.4f}")

st.markdown('---')
st.header("🔍 Analysis")

col1, col2, col3 = st.columns(3)


hide_bond_weights = st.sidebar.checkbox("Hide bond weights", help="Toggle bond weight visualization")
hide_atom_weights = st.sidebar.checkbox("Hide atom weights", help="Toggle atom weight visualization")

hide_bond_weights=False
png_frag_attn, png_frag_highlight, frag_w, connection_w, atoms_in_frags = viz.frag_weight_highlight()
png_attr, attr_atom_weights, frag_contributions = get_attr_image(selected, model_config, chkpt_path, prop_type)

def highlight_contribution(val):
    if isinstance(val, (int, float)):
        color = 'lightcoral' if val < 0 else 'lightblue'
        return f'background-color: {color}'
    return ''

def show_contrib_table(df, contrib_col, display_cols, label_cols):
    df_display = df[display_cols].copy()
    df_display['abs_attr'] = df_display[contrib_col].abs()
    df_display = df_display.sort_values('abs_attr', ascending=False).drop('abs_attr', axis=1)
    df_display.columns = label_cols
    st.dataframe(
        df_display.style.applymap(highlight_contribution, subset=['Contribution']),
        hide_index=True,
        use_container_width=True,
        height=350
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean", f"{df[contrib_col].mean():.4f}")
    m2.metric("Max", f"{df[contrib_col].max():.4f}")
    m3.metric("Min", f"{df[contrib_col].min():.4f}")

try:
    with st.spinner("🔄 Calculating contributions..."):
        df_atom_contrib, df_bond_contrib, df_fbond_contrib = viz.get_all_contributions(prop_type)

    tab_atoms, tab_bonds, tab_frags, tab_fconn = st.tabs([
        "⚛️ Atoms", "🔗 Bonds", "🧩 Fragments", "🔀 Fragment Connections"
    ])

    with tab_atoms:
        v_col, t_col = st.columns(2)
        with v_col:
            st.subheader("Atom Weights")
            png_atoms, atom_weights = viz.vizualize_atom_weights(True, False)
            st.image(png_atoms, use_column_width=True)
            with st.expander("📊 View Atom Weight Values", expanded=False):
                attn_atoms = pd.DataFrame(atom_weights)
                attn_atoms.index.rename('Atom Index', inplace=True)
                attn_atoms.columns = ['Weight']
                st.dataframe(attn_atoms, use_container_width=True)
        with t_col:
            st.subheader("Atom Contributions")
            st.markdown("Impact of each **atom** on the prediction (masking).")
            show_contrib_table(df_atom_contrib, 'attr',
                               ['atom_index', 'atom_type', 'attr'],
                               ['Atom Index', 'Symbol', 'Contribution'])

    with tab_bonds:
        v_col, t_col = st.columns(2)
        with v_col:
            st.subheader("Bond Weights")
            png_bonds, bond_atom_weights = viz.vizualize_atom_weights(False, True)
            st.image(png_bonds, use_column_width=True)
            with st.expander("📊 View Bond Weight Values", expanded=False):
                bond_atoms = pd.DataFrame(bond_atom_weights)
                bond_atoms.index.rename('Atom Index', inplace=True)
                bond_atoms.columns = ['Weight']
                st.dataframe(bond_atoms, use_container_width=True)
        with t_col:
            st.subheader("Bond Contributions")
            st.markdown("Impact of each **bond** on the prediction (masking).")
            if not df_bond_contrib.empty:
                show_contrib_table(df_bond_contrib, 'attr',
                                   ['bond_index', 'begin_atom', 'end_atom', 'attr'],
                                   ['Bond Index', 'Begin Atom', 'End Atom', 'Contribution'])
            else:
                st.info("No bonds to analyze.")

    with tab_frags:
        # Row 1: Fragment decomposition + atom mapping
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("Fragment Decomposition")
            st.image(png_frag_highlight, use_column_width=True)
            st.image(png_frag_attn, use_column_width=True)
            st.caption("Attention-based fragment weights")
            with st.expander("📊 View Fragment Weight Values", expanded=False):
                st.dataframe(frag_w, use_container_width=True)
        with row1_col2:
            st.subheader("Fragment Atom Mapping")
            with st.expander("📋 View Fragment Atom Mapping", expanded=True):
                df_atoms_in_frags = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in atoms_in_frags.items()])).T
                df_atoms_in_frags.index.rename('Fragment', inplace=True)
                st.dataframe(df_atoms_in_frags, use_container_width=True)

        st.markdown('---')

        # Row 2: Fragment attribution image + contribution table
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.subheader("Fragment Contributions (Visual)")
            st.image(png_attr, use_column_width=True)
            st.caption("Fragment contributions via masking-based attribution")
        with row2_col2:
            st.subheader("Fragment Contributions (Table)")
            st.markdown("Impact of each **fragment** on the prediction (masking).")
            df_frag_contrib = pd.DataFrame(frag_contributions)
            df_frag_contrib['atoms'] = df_frag_contrib['atoms'].apply(lambda x: ', '.join(str(a) for a in x))
            show_contrib_table(df_frag_contrib, 'contribution',
                               ['fragment_index', 'atoms', 'contribution'],
                               ['Fragment Index', 'Atoms', 'Contribution'])

    with tab_fconn:
        v_col, t_col = st.columns(2)
        with v_col:
            st.subheader("Connection Weight Values")
            with st.expander("🔗 View Connection Weight Values", expanded=True):
                st.dataframe(connection_w, use_container_width=True)
        with t_col:
            st.subheader("Fragment Connection Contributions")
            st.markdown("Impact of **inter-fragment connections** on the prediction (masking).")
            if not df_fbond_contrib.empty:
                show_contrib_table(df_fbond_contrib, 'attr',
                                   ['fragbond_node_index', 'begin_index', 'end_index', 'attr'],
                                   ['Connection Index', 'Begin Fragment', 'End Fragment', 'Contribution'])
            else:
                st.info("Single fragment molecule — no inter-fragment connections.")

except Exception as e:
    st.error(f"❌ Error calculating contributions: {str(e)}")
    st.exception(e)

# Footer section
st.markdown('---')
st.sidebar.markdown('---')
st.sidebar.info(f"**Current Molecule:** `{selected}`")